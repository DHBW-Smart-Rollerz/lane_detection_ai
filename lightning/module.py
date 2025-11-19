from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch

from utils.common import calc_loss, get_model, inference
from utils.dist_utils import dist_print
from utils.factory import (
    get_loss_dict,
    get_metric_dict,
    get_optimizer,
    get_scheduler,
)
from utils.metrics import reset_metrics, update_metrics


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
        self.loss_dict = get_loss_dict(cfg)
        self.metric_dict = get_metric_dict(cfg)
        self.metric_log_interval = getattr(cfg, "metric_log_interval", 20)
        self.lr_scheduler = None
        self._iters_per_epoch = None
        self.logger_adapter = _LightningLoggerAdapter(self)

        if getattr(cfg, "finetune", None):
            self._load_finetune_weights(cfg.finetune)
        elif getattr(cfg, "resume", None):
            self._load_resume_weights(cfg.resume)

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
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ) -> None:
        super().optimizer_step(
            epoch,
            batch_idx,
            optimizer,
            optimizer_closure,
            on_tpu,
            using_native_amp,
            using_lbfgs,
        )
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(self.global_step)

    def on_train_epoch_start(self) -> None:
        reset_metrics(self.metric_dict)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # DALI already provides GPU tensors, so we bypass Lightning's device transfer.
        return batch

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Avoid overwriting cfg when resuming via Lightning checkpoints.
        checkpoint_cfg = checkpoint.get("cfg")
        if checkpoint_cfg is not None:
            self.cfg = checkpoint_cfg

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["cfg"] = self.cfg
