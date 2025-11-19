from typing import Optional

import pytorch_lightning as pl

from utils.common import get_train_loader


class SmartrollerzDataModule(pl.LightningDataModule):
    """Lightning wrapper around the existing DALI-based TrainCollect loader."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._train_loader = None
        self.train_loader_len = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in (None, "fit"):
            return
        if self._train_loader is None:
            self._train_loader = get_train_loader(self.cfg)
            self.train_loader_len = len(self._train_loader)

    def train_dataloader(self):
        if self._train_loader is None:
            self.setup("fit")
        # Reset DALI iterator so each epoch starts from the beginning.
        if hasattr(self._train_loader, "reset"):
            self._train_loader.reset()
        return self._train_loader

    def teardown(self, stage: Optional[str] = None) -> None:
        if stage not in (None, "fit"):
            return
        if self._train_loader is not None and hasattr(self._train_loader, "reset"):
            self._train_loader.reset()
