import os
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
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

    checkpoint_callback = ModelCheckpoint(
        dirpath=work_dir,
        save_last=True,
        save_top_k=-1,
        every_n_epochs=max(1, getattr(cfg, "save_every_n_epochs", 50)),
        filename="epoch{epoch}",
    )
    callbacks = [checkpoint_callback]
    best_metric_name: Optional[str] = getattr(cfg, "best_metric", "val/local_f1")
    best_metric_mode = getattr(cfg, "best_metric_mode", "max")
    best_checkpoint_callback: Optional[ModelCheckpoint] = None
    if has_validation and best_metric_name:
        best_checkpoint_callback = ModelCheckpoint(
            dirpath=work_dir,
            filename="best-{epoch}",
            monitor=best_metric_name,
            mode=best_metric_mode,
            save_top_k=1,
            save_last=False,
            auto_insert_metric_name=True,
        )
        callbacks.append(best_checkpoint_callback)
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
    final_ckpt = os.path.join(work_dir, "final.ckpt")
    trainer.save_checkpoint(final_ckpt)

    best_ckpt_path: Optional[str] = None
    best_metric_score: Optional[float] = None
    if best_checkpoint_callback is not None and best_checkpoint_callback.best_model_path:
        best_ckpt_path = best_checkpoint_callback.best_model_path
        if best_checkpoint_callback.best_model_score is not None:
            best_metric_score = float(best_checkpoint_callback.best_model_score.cpu().item())
    elif os.path.exists(final_ckpt):
        best_ckpt_path = final_ckpt

    if mlflow_logger is not None:
        if best_metric_score is not None and best_metric_name:
            metric_tag = f"best/{best_metric_name.replace('/', '_')}"
            mlflow_logger.experiment.log_param(
                mlflow_logger.run_id,
                metric_tag,
                float(best_metric_score),
            )
        artifact_path = getattr(cfg, "mlflow_best_model_artifact_path", "checkpoints")
        if artifact_path == "":
            artifact_path = None
        if best_ckpt_path and os.path.exists(best_ckpt_path):
            mlflow_logger.experiment.log_artifact(
                mlflow_logger.run_id,
                best_ckpt_path,
                artifact_path=artifact_path,
            )


if __name__ == "__main__":
    main()
