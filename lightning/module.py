from typing import Any, Dict, Optional, Tuple

import os

import pytorch_lightning as pl
import torch

from utils.common import calc_loss, get_model, inference
from pytorch_lightning.loggers import MLFlowLogger
from utils.dist_utils import dist_print
from utils.factory import (
    get_loss_dict,
    get_metric_dict,
    get_optimizer,
    get_scheduler,
)
from utils.metrics import reset_metrics, update_metrics
from utils.smartrollerz_metric import UFLDV2F1Score
from evaluation.eval_wrapper import eval_lane


def _compute_f1_from_counts(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


class _LightningLoggerAdapter:
    """Adapts LightningModule logging API to the legacy logger interface."""

    def __init__(self, module: "LaneDetectionLightningModule") -> None:
        self.module = module
        self.batch_size: Optional[int] = None

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def add_scalar(self, tag: str, scalar_value: Any, global_step: Optional[int] = None) -> None:
        # Lightning handles the step indexing internally, but we forward the scalar for parity
        self.module.log(
            tag,
            scalar_value,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            batch_size=self.batch_size,
        )


class LaneDetectionLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = get_model(cfg)
        self._finetune_active = bool(getattr(cfg, "finetune", None) or getattr(cfg, "finetune_active", False))
        self._finetune_stagewise_enabled = bool(getattr(cfg, "finetune_stagewise", True)) and self._finetune_active
        self._finetune_discriminative_lr = bool(getattr(cfg, "finetune_discriminative_lr", True)) and self._finetune_active
        self._head_warmup_epochs = max(0, int(getattr(cfg, "finetune_head_warmup_epochs", 0) or 0))
        self._backbone_prefix = str(getattr(cfg, "finetune_backbone_prefix", "model."))
        self._finetune_freeze_state = None
        self.loss_dict = get_loss_dict(cfg)
        self.metric_dict = get_metric_dict(cfg)
        self.val_metric_dict = get_metric_dict(cfg)
        self.metric_log_interval = getattr(cfg, "metric_log_interval", 20)
        self.lr_scheduler = None
        self._iters_per_epoch = None
        self.logger_adapter = _LightningLoggerAdapter(self)
        self._last_val_f1: Optional[float] = None
        self._last_val_precision: Optional[float] = None
        self._last_val_recall: Optional[float] = None
        self._last_val_tp: Optional[float] = None
        self._last_val_fp: Optional[float] = None
        self._last_val_fn: Optional[float] = None
        try:
            self.val_f1_metric: Optional[UFLDV2F1Score] = UFLDV2F1Score(cfg)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.val_f1_metric = None
            dist_print(f"[Lightning] Failed to initialize UFLDV2F1Score: {exc}")

        if getattr(cfg, "finetune", None):
            self._load_finetune_weights(cfg.finetune)
        elif getattr(cfg, "resume", None):
            self._load_resume_weights(cfg.resume)

        # If stage-wise finetuning is enabled, apply initial freeze before optimizer creation.
        self._maybe_update_finetune_freeze_state(epoch=0, force=True)

        self.save_hyperparameters({"note": getattr(cfg, "note", ""), "dataset": cfg.dataset})

    def _load_finetune_weights(self, ckpt_path: str) -> None:
        dist_print("[Lightning] Finetune from", ckpt_path)
        state_all = torch.load(ckpt_path, map_location="cpu").get("model", {})
        state_clip = {k: v for k, v in state_all.items() if "model" in k}
        self.model.load_state_dict(state_clip, strict=False)

    def _load_resume_weights(self, ckpt_path: str) -> None:
        dist_print("[Lightning] Resume weights from", ckpt_path)
        state = torch.load(ckpt_path, map_location="cpu")
        if "model" in state:
            self.model.load_state_dict(state["model"], strict=False)

    def set_iters_per_epoch(self, iters: int) -> None:
        self._iters_per_epoch = max(1, iters)

    def _split_backbone_head_params(self):
        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if name.startswith(self._backbone_prefix):
                backbone_params.append(param)
            else:
                head_params.append(param)
        return backbone_params, head_params

    def _set_backbone_requires_grad(self, enabled: bool) -> None:
        changed = 0
        for name, param in self.model.named_parameters():
            if name.startswith(self._backbone_prefix):
                if param.requires_grad != enabled:
                    param.requires_grad = enabled
                    changed += 1
        state_txt = "trainable" if enabled else "frozen"
        dist_print(f"[Lightning] Backbone set to {state_txt} (changed_params={changed})")

    def _maybe_update_finetune_freeze_state(self, epoch: int, force: bool = False) -> None:
        if not self._finetune_stagewise_enabled:
            return

        freeze_backbone = epoch < self._head_warmup_epochs
        if not force and self._finetune_freeze_state == freeze_backbone:
            return

        self._set_backbone_requires_grad(enabled=not freeze_backbone)
        self._finetune_freeze_state = freeze_backbone
        if freeze_backbone:
            dist_print(
                f"[Lightning] Stage-wise finetune: head warmup active at epoch={epoch} "
                f"(warmup_epochs={self._head_warmup_epochs})."
            )
        else:
            dist_print(f"[Lightning] Stage-wise finetune: backbone unfrozen at epoch={epoch}.")

    def forward(self, batch: Dict[str, torch.Tensor]):
        return self.model(batch["images"])

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        batch_size = batch["images"].shape[0]
        self.logger_adapter.set_batch_size(batch_size)

        results = inference(self.model, batch, self.cfg.dataset)
        loss = calc_loss(
            self.loss_dict,
            results,
            self.logger_adapter,
            global_step=self.global_step,
            epoch=self.current_epoch,
        )
        self.log("train/loss", loss, on_step=True, prog_bar=True, batch_size=batch_size)

        if (self.global_step % self.metric_log_interval) == 0:
            reset_metrics(self.metric_dict)
            update_metrics(self.metric_dict, results)
            for me_name, me_op in zip(self.metric_dict["name"], self.metric_dict["op"]):
                self.log(
                    f"metric/{me_name}",
                    me_op.get(),
                    on_step=True,
                    prog_bar=False,
                    batch_size=batch_size,
                )
            opt = self.optimizers(use_pl_optimizer=False)
            if opt is not None:
                self.log(
                    "meta/lr",
                    opt.param_groups[0]["lr"],
                    on_step=True,
                    prog_bar=False,
                    batch_size=batch_size,
                )
        return loss

    def configure_optimizers(self):
        use_discriminative = self._finetune_discriminative_lr
        if use_discriminative:
            backbone_params, head_params = self._split_backbone_head_params()
            if len(backbone_params) == 0 or len(head_params) == 0:
                dist_print(
                    "[Lightning] Discriminative LR disabled: could not split backbone/head params "
                    f"with prefix '{self._backbone_prefix}'. Falling back to default optimizer."
                )
                optimizer = get_optimizer(self.model, self.cfg)
            else:
                base_lr = float(self.cfg.learning_rate)
                backbone_lr_scale = float(getattr(self.cfg, "finetune_backbone_lr_scale", 0.1))
                head_lr_scale = float(getattr(self.cfg, "finetune_head_lr_scale", 1.0))
                backbone_lr = base_lr * backbone_lr_scale
                head_lr = base_lr * head_lr_scale

                param_groups = [
                    {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
                    {"params": head_params, "lr": head_lr, "name": "head"},
                ]
                if self.cfg.optimizer == "Adam":
                    optimizer = torch.optim.Adam(param_groups, weight_decay=self.cfg.weight_decay)
                elif self.cfg.optimizer == "SGD":
                    optimizer = torch.optim.SGD(
                        param_groups,
                        momentum=self.cfg.momentum,
                        weight_decay=self.cfg.weight_decay,
                    )
                else:
                    raise NotImplementedError
                dist_print(
                    "[Lightning] Using discriminative LR groups "
                    f"(base_lr={base_lr:.3e}, backbone_lr={backbone_lr:.3e}, head_lr={head_lr:.3e})."
                )
        else:
            optimizer = get_optimizer(self.model, self.cfg)
        if self._iters_per_epoch is None:
            raise RuntimeError("iters_per_epoch is not set. Call set_iters_per_epoch() before training.")
        self.lr_scheduler = get_scheduler(optimizer, self.cfg, self._iters_per_epoch)
        return optimizer

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure=None,
    ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(self.global_step)

    def on_train_epoch_start(self) -> None:
        self._maybe_update_finetune_freeze_state(epoch=self.current_epoch)
        reset_metrics(self.metric_dict)

    def on_validation_epoch_start(self) -> None:
        reset_metrics(self.val_metric_dict)
        if self.val_f1_metric is not None:
            self.val_f1_metric.reset()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # DALI already provides GPU tensors, so we bypass Lightning's device transfer.
        return batch

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        if batch is None:
            return None
        batch_size = batch["images"].shape[0]
        results = inference(self.model, batch, self.cfg.dataset)
        loss = calc_loss(
            self.loss_dict,
            results,
            logger=None,
            global_step=self.global_step,
            epoch=self.current_epoch,
        )
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        update_metrics(self.val_metric_dict, results)
        if self.val_f1_metric is not None:
            self.val_f1_metric.update(results)
        return loss

    def on_validation_epoch_end(self) -> None:
        if getattr(self.cfg, "eval_during_training", False):
            self._update_external_eval_metrics()

        precision = self._last_val_precision
        recall = self._last_val_recall
        f1 = self._last_val_f1
        tp = self._last_val_tp
        fp = self._last_val_fp
        fn = self._last_val_fn

        for idx, (me_name, me_op) in enumerate(zip(self.val_metric_dict["name"], self.val_metric_dict["op"])):
            self.log(
                f"val/{me_name}",
                me_op.get(),
                on_step=False,
                on_epoch=True,
                prog_bar=idx == 0,
            )

        if f1 is not None:
            self.log("val/precision", precision, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/recall", recall, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/f1", f1, on_step=False, on_epoch=True, prog_bar=True)
        if tp is not None:
            self.log("val/tp", tp, on_step=False, on_epoch=True, prog_bar=False)
        if fp is not None:
            self.log("val/fp", fp, on_step=False, on_epoch=True, prog_bar=False)
        if fn is not None:
            self.log("val/fn", fn, on_step=False, on_epoch=True, prog_bar=False)

        self._last_val_precision = None
        self._last_val_recall = None
        self._last_val_f1 = None
        self._last_val_tp = None
        self._last_val_fp = None
        self._last_val_fn = None

        if self.val_f1_metric is not None:
            metric_values = self.val_f1_metric.compute()
            self.val_f1_metric.reset()
            if metric_values:
                self.log(
                    "val/local_precision",
                    metric_values["precision"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    "val/local_recall",
                    metric_values["recall"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    "val/local_f1",
                    metric_values["f1"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                # Also explicitly log the same metrics to MLflow with the epoch as the step.
                # This makes sure MLflow shows one metric point per validation epoch
                # (some logger backends omit the epoch step), and helps downstream
                # automation pick the best epoch by comparing metric steps.
                try:
                    trainer = getattr(self, "trainer", None)
                    if trainer is not None and hasattr(trainer, "loggers"):
                        for lg in trainer.loggers:
                            if isinstance(lg, MLFlowLogger):
                                run_id = lg.run_id
                                # log_metric expects (run_id, key, value, step)
                                lg.experiment.log_metric(run_id, "val/local_f1", float(metric_values["f1"]), step=int(self.current_epoch))
                                lg.experiment.log_metric(run_id, "val/local_precision", float(metric_values["precision"]), step=int(self.current_epoch))
                                lg.experiment.log_metric(run_id, "val/local_recall", float(metric_values["recall"]), step=int(self.current_epoch))
                except Exception:
                    # Non-critical: if MLflow isn't available or logging fails, continue silently
                    pass
                self.log(
                    "val/local_tp",
                    metric_values["tp"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    "val/local_fp",
                    metric_values["fp"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    "val/local_fn",
                    metric_values["fn"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
        

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Avoid overwriting cfg when resuming via Lightning checkpoints.
        checkpoint_cfg = checkpoint.get("cfg")
        if checkpoint_cfg is not None:
            self.cfg = checkpoint_cfg

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["cfg"] = self.cfg
    
    def _has_external_eval_inputs(self) -> bool:
        dataset = getattr(self.cfg, "dataset", "")
        data_root = getattr(self.cfg, "data_root", "")
        if not data_root:
            return False
        if dataset == "Smartrollerz":
            required = [
                os.path.join(data_root, "test.txt"),
                os.path.join(data_root, "test_label.json"),
            ]
            missing = [path for path in required if not os.path.exists(path)]
            if missing:
                dist_print(
                    "[Lightning] External evaluation disabled: missing Smartrollerz test files: "
                    + ", ".join(missing)
                )
                return False
        return True
    
    def _ensure_test_work_dir(self) -> str:
        test_dir = getattr(self.cfg, "test_work_dir", None)
        if not test_dir:
            base_dir = None
            if getattr(self, "trainer", None) is not None:
                base_dir = self.trainer.default_root_dir
                if not base_dir:
                    base_dir = getattr(self.trainer, "log_dir", None)
            if not base_dir:
                base_dir = getattr(self.cfg, "log_path", os.getcwd())
            test_dir = os.path.join(base_dir, "eval_tmp")
            self.cfg.test_work_dir = test_dir
        os.makedirs(test_dir, exist_ok=True)
        return test_dir

    def _compute_external_eval_metrics_rank_zero(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        if not getattr(self.cfg, "eval_during_training", False):
            return None
        if getattr(self, "trainer", None) is None or getattr(self.trainer, "sanity_checking", False):
            return None
        if not getattr(self.trainer, "is_global_zero", True):
            return None
        if not self._has_external_eval_inputs():
            return None

        self._ensure_test_work_dir()
        original_distributed = getattr(self.cfg, "distributed", False)
        self.cfg.distributed = False
        was_training = self.model.training
        metrics_tuple: Optional[Tuple[float, float, float, float, float, float]] = None
        try:
            try:
                eval_result = eval_lane(self.model, self.cfg, ep=self.current_epoch, logger=None, return_counts=True)
            except Exception as exc:
                dist_print(f"[Lightning] External evaluation failed: {exc}")
                eval_result = None
            if eval_result is not None:
                _, summary = eval_result
                tp = float(summary.get('tp', 0.0))
                fp = float(summary.get('fp', 0.0))
                fn = float(summary.get('fn', 0.0))
                precision, recall, f1 = _compute_f1_from_counts(tp, fp, fn)
                metrics_tuple = (precision, recall, f1, tp, fp, fn)
        finally:
            self.cfg.distributed = original_distributed
            self.model.train(was_training)
        return metrics_tuple

    def _gather_eval_metrics(self, metrics_tuple: Optional[Tuple[float, float, float, float, float, float]]) -> Optional[Tuple[float, float, float, float, float, float]]:
        device = getattr(self, "device", torch.device("cpu"))
        if not isinstance(device, torch.device):
            device = torch.device(str(device))
        payload = torch.zeros(7, dtype=torch.float32, device=device)
        if metrics_tuple is not None:
            payload[:6] = torch.tensor(metrics_tuple, dtype=torch.float32, device=device)
            payload[6] = 1.0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = self.all_gather(payload)
            payload = gathered[0]
        if payload[6].item() < 0.5:
            return None
        values = payload[:6].tolist()
        return tuple(values)  # type: ignore[return-value]

    def _update_external_eval_metrics(self) -> None:
        metrics = self._gather_eval_metrics(self._compute_external_eval_metrics_rank_zero())
        if metrics is None:
            self._last_val_precision = None
            self._last_val_recall = None
            self._last_val_f1 = None
            self._last_val_tp = None
            self._last_val_fp = None
            self._last_val_fn = None
            return

        precision, recall, f1, tp, fp, fn = metrics
        self._last_val_precision = precision
        self._last_val_recall = recall
        self._last_val_f1 = f1
        self._last_val_tp = tp
        self._last_val_fp = fp
        self._last_val_fn = fn
