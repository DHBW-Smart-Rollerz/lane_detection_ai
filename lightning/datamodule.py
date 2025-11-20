from typing import Optional

import pytorch_lightning as pl

from utils.common import get_eval_loader, get_train_loader


class SmartrollerzDataModule(pl.LightningDataModule):
    """Lightning wrapper around the existing DALI-based TrainCollect loader."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._train_loader = None
        self._val_loader = None
        self.train_loader_len = None
        self.val_loader_len = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            if self._train_loader is None:
                self._train_loader = get_train_loader(self.cfg)
                self.train_loader_len = len(self._train_loader)
        if stage in (None, "validate"):
            if self._val_loader is None:
                self._val_loader = get_eval_loader(self.cfg)
                if self._val_loader is not None:
                    self.val_loader_len = len(self._val_loader)

    def train_dataloader(self):
        if self._train_loader is None:
            self.setup("fit")
        # Reset DALI iterator so each epoch starts from the beginning.
        if hasattr(self._train_loader, "reset"):
            self._train_loader.reset()
        return self._train_loader

    def val_dataloader(self):
        self.setup("validate")
        if self._val_loader is None:
            return None
        if hasattr(self._val_loader, "reset"):
            self._val_loader.reset()
        return self._val_loader

    def teardown(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            if self._train_loader is not None and hasattr(self._train_loader, "reset"):
                self._train_loader.reset()
        if stage in (None, "validate"):
            if self._val_loader is not None and hasattr(self._val_loader, "reset"):
                self._val_loader.reset()

    @property
    def has_validation_loader(self) -> bool:
        return self._val_loader is not None
