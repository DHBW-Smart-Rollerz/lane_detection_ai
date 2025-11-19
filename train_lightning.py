import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

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

    model = LaneDetectionLightningModule(cfg)
    if datamodule.train_loader_len is None:
        raise RuntimeError("DataModule failed to initialize train loader length.")
    model.set_iters_per_epoch(datamodule.train_loader_len)

    checkpoint_callback = ModelCheckpoint(
        dirpath=work_dir,
        save_last=True,
        save_top_k=-1,
        every_n_epochs=max(1, getattr(cfg, "save_every_n_epochs", 10)),
        filename="epoch{epoch}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger(save_dir=work_dir, name="events")

    trainer = pl.Trainer(
        max_epochs=cfg.epoch,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
        default_root_dir=work_dir,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=getattr(cfg, "log_interval", 20),
        gradient_clip_val=getattr(cfg, "grad_clip", 0.0) or None,
        deterministic=getattr(cfg, "deterministic", False),
    )

    ckpt_path = None
    if getattr(cfg, "resume", None) and cfg.resume.endswith(".ckpt"):
        ckpt_path = cfg.resume

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    final_ckpt = os.path.join(work_dir, "final.ckpt")
    trainer.save_checkpoint(final_ckpt)


if __name__ == "__main__":
    main()
