import os
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger

from lightning.datamodule import SmartrollerzDataModule
from lightning.module import LaneDetectionLightningModule
from utils.common import get_work_dir, merge_config


class _ForceLoggerFlushCallback(Callback):
    """Ensures logger backends flush their internal buffers during training."""

    def _iter_loggers(self, trainer: pl.Trainer):
        if trainer is None:
            return []
        if getattr(trainer, "loggers", None):
            loggers = trainer.loggers
        else:
            logger = getattr(trainer, "logger", None)
            loggers = [logger] if logger is not None else []
        if isinstance(loggers, (list, tuple)):
            return list(loggers)
        try:
            return list(loggers)
        except TypeError:
            return [loggers]

    def _flush(self, trainer: pl.Trainer) -> None:
        for logger in self._iter_loggers(trainer):
            if logger is None:
                continue
            experiment = getattr(logger, "experiment", None)
            if experiment is None:
                continue
            flush_fn = getattr(experiment, "flush", None)
            if callable(flush_fn):
                try:
                    flush_fn()
                except Exception:
                    pass

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._flush(trainer)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._flush(trainer)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._flush(trainer)


def _save_model_artifacts(pl_module: pl.LightningModule, weights_path: str, model_path: Optional[str] = None) -> None:
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    torch.save({"model": pl_module.model.state_dict()}, weights_path)
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(pl_module.model, model_path)


class _BestPthSaverCallback(Callback):
    def __init__(self, monitor: str, mode: str, weights_path: str, model_path: Optional[str] = None):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.weights_path = weights_path
        self.model_path = model_path
        self.best_score: Optional[float] = None

    def _is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score
        return score > self.best_score

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        score = float(metric.detach().cpu().item() if torch.is_tensor(metric) else metric)
        if self._is_better(score):
            self.best_score = score
            _save_model_artifacts(pl_module, self.weights_path, self.model_path)


def main():
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()
    work_dir = get_work_dir(cfg)
    os.makedirs(work_dir, exist_ok=True)

    datamodule = SmartrollerzDataModule(cfg)
    datamodule.setup("fit")
    datamodule.setup("validate")

    has_validation = datamodule.has_validation_loader
    limit_val_batches = 1.0 if has_validation else 0.0
    if not has_validation:
        print("[Lightning] No eval_gt split detected – validation during training is disabled.")

    model = LaneDetectionLightningModule(cfg)
    if datamodule.train_loader_len is None:
        raise RuntimeError("DataModule failed to initialize train loader length.")
    model.set_iters_per_epoch(datamodule.train_loader_len)

    callbacks = []
    best_metric_name: Optional[str] = getattr(cfg, "best_metric", "val/local_f1")
    best_metric_mode = getattr(cfg, "best_metric_mode", "max")
    best_weights_path = os.path.join(work_dir, "best_weights.pth")
    best_model_path = os.path.join(work_dir, "best_model.pth")
    best_saver_callback: Optional[_BestPthSaverCallback] = None
    if has_validation and best_metric_name:
        best_saver_callback = _BestPthSaverCallback(
            monitor=best_metric_name,
            mode=best_metric_mode,
            weights_path=best_weights_path,
            model_path=best_model_path,
        )
        callbacks.append(best_saver_callback)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    tensorboard_logger = TensorBoardLogger(
        save_dir=work_dir,
        name="events",
        default_hp_metric=False,
        flush_secs=getattr(cfg, "tensorboard_flush_secs", 5),
        max_queue=getattr(cfg, "tensorboard_max_queue", 1),
    )

    loggers = [tensorboard_logger]
    mlflow_uri = getattr(cfg, "mlflow_uri", None)
    mlflow_experiment = getattr(cfg, "mlflow_experiment", None)
    mlflow_run_name = getattr(cfg, "mlflow_run_name", None)
    mlflow_logger = None
    if mlflow_uri or mlflow_experiment or mlflow_run_name:
        mlflow_logger = MLFlowLogger(
            experiment_name=mlflow_experiment or "lane-detection",
            tracking_uri=mlflow_uri,
            run_name=mlflow_run_name or os.path.basename(work_dir),
        )
        used_config_path = os.path.abspath(args.config)
        used_config_name = os.path.basename(used_config_path)
        mlflow_logger.log_hyperparams(
            {
                "dataset": getattr(cfg, "dataset", "unknown"),
                "backbone": getattr(cfg, "backbone", "unknown"),
                "epoch": getattr(cfg, "epoch", 0),
                "batch_size": getattr(cfg, "batch_size", 0),
                "learning_rate": getattr(cfg, "learning_rate", 0.0),
                "note": getattr(cfg, "note", ""),
                "config_file": used_config_name,
                "config_path": used_config_path,
            }
        )
        if os.path.isfile(used_config_path):
            mlflow_logger.experiment.log_artifact(
                mlflow_logger.run_id,
                used_config_path,
                artifact_path="config",
            )
        loggers.append(mlflow_logger)

    callbacks.append(lr_monitor)
    callbacks.append(_ForceLoggerFlushCallback())

    trainer = pl.Trainer(
        max_epochs=cfg.epoch,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        enable_checkpointing=False,
        reload_dataloaders_every_n_epochs=1,
        default_root_dir=work_dir,
        logger=loggers if len(loggers) > 1 else loggers[0],
        callbacks=callbacks,
        log_every_n_steps=getattr(cfg, "log_interval", 20),
        gradient_clip_val=getattr(cfg, "grad_clip", 0.0) or None,
        deterministic=getattr(cfg, "deterministic", False),
        limit_val_batches=limit_val_batches,
    )

    ckpt_path = None
    if getattr(cfg, "resume", None) and cfg.resume.endswith(".ckpt"):
        ckpt_path = cfg.resume

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    for lg in loggers:
        experiment = getattr(lg, "experiment", None)
        if experiment is None:
            continue
        flush_fn = getattr(experiment, "flush", None)
        if callable(flush_fn):
            try:
                flush_fn()
            except Exception:
                pass
        close_fn = getattr(experiment, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
    final_weights_path = os.path.join(work_dir, "final_weights.pth")
    final_model_path = os.path.join(work_dir, "final_model.pth")
    if trainer.is_global_zero:
        _save_model_artifacts(model, final_weights_path, final_model_path)

    best_artifact_weights_path: Optional[str] = None
    best_artifact_model_path: Optional[str] = None
    best_metric_score: Optional[float] = None
    if best_saver_callback is not None and os.path.exists(best_weights_path):
        best_artifact_weights_path = best_weights_path
        if os.path.exists(best_model_path):
            best_artifact_model_path = best_model_path
        if best_saver_callback.best_score is not None:
            best_metric_score = float(best_saver_callback.best_score)
    elif os.path.exists(final_weights_path):
        best_artifact_weights_path = final_weights_path
        if os.path.exists(final_model_path):
            best_artifact_model_path = final_model_path

    if mlflow_logger is not None:
        if best_metric_score is not None and best_metric_name:
            metric_tag = f"best/{best_metric_name.replace('/', '_')}"
            mlflow_logger.experiment.log_param(
                mlflow_logger.run_id,
                metric_tag,
                float(best_metric_score),
            )
        artifact_path = getattr(cfg, "mlflow_best_model_artifact_path", "models")
        if artifact_path == "":
            artifact_path = None
        if best_artifact_weights_path and os.path.exists(best_artifact_weights_path):
            mlflow_logger.experiment.log_artifact(
                mlflow_logger.run_id,
                best_artifact_weights_path,
                artifact_path=artifact_path,
            )
        if best_artifact_model_path and os.path.exists(best_artifact_model_path):
            mlflow_logger.experiment.log_artifact(
                mlflow_logger.run_id,
                best_artifact_model_path,
                artifact_path=artifact_path,
            )


if __name__ == "__main__":
    main()
