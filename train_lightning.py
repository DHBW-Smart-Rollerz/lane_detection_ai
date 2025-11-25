import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger

from lightning.datamodule import SmartrollerzDataModule
from lightning.module import LaneDetectionLightningModule
from utils.common import get_work_dir, merge_config


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
    lr_monitor = LearningRateMonitor(logging_interval="step")
    tensorboard_logger = TensorBoardLogger(save_dir=work_dir, name="events")

    loggers = [tensorboard_logger]
    mlflow_uri = getattr(cfg, "mlflow_uri", None)
    mlflow_experiment = getattr(cfg, "mlflow_experiment", None)
    mlflow_run_name = getattr(cfg, "mlflow_run_name", None)
    mlflow_logger = None
    if mlflow_uri or mlflow_experiment or mlflow_run_name:
        mlflow_logger = MLFlowLogger(
            experiment_name=mlflow_experiment or "lane-detection",
            tracking_uri=mlflow_uri,mlflow_logger
            run_name=mlflow_run_name or os.path.basename(work_dir),
        )
        mlflow_logger.log_hyperparams(
            {
                "dataset": getattr(cfg, "dataset", "unknown"),
                "backbone": getattr(cfg, "backbone", "unknown"),
                "epoch": getattr(cfg, "epoch", 0),
                "batch_size": getattr(cfg, "batch_size", 0),
                "learning_rate": getattr(cfg, "learning_rate", 0.0),
                "note": getattr(cfg, "note", ""),
            }
        )
        loggers.append(mlflow_logger)

    trainer = pl.Trainer(
        max_epochs=cfg.epoch,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        reload_dataloaders_every_n_epochs=1,
        default_root_dir=work_dir,
    logger=loggers if len(loggers) > 1 else loggers[0],
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=getattr(cfg, "log_interval", 20),
        gradient_clip_val=getattr(cfg, "grad_clip", 0.0) or None,
        deterministic=getattr(cfg, "deterministic", False),
        limit_val_batches=limit_val_batches,
    )

    ckpt_path = None
    if getattr(cfg, "resume", None) and cfg.resume.endswith(".ckpt"):
        ckpt_path = cfg.resume

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    final_ckpt = os.path.join(work_dir, "final.ckpt")
    trainer.save_checkpoint(final_ckpt)


if __name__ == "__main__":
    main()
