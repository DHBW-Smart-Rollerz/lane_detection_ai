"""Loading of the python model config files (subset of the mmcv Config)."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from addict import Dict

if TYPE_CHECKING:
    from collections.abc import Iterator

BASE_KEY = "_base_"
DELETE_KEY = "_delete_"


class ConfigDict(Dict):
    """Dict with attribute access that raises for missing keys."""

    def __missing__(self, name: str) -> Any:
        raise KeyError(name)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except KeyError:
            msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
            raise AttributeError(msg) from None


class Config:
    """Config loaded from a python file, accessible as attributes or items.

    A config file can inherit from other config files by setting `_base_` to a
    path (or a list of paths) relative to itself. Values of the file overwrite the
    values of its bases; nested dicts are merged unless they set `_delete_=True`.

    Example:
        >>> cfg = Config.fromfile("config/model_config.py")
        >>> cfg.backbone
        '18'
        >>> getattr(cfg, "missing_key", "default")
        'default'
    """

    def __init__(
        self, cfg_dict: dict | None = None, filename: str | None = None
    ) -> None:
        """Initialize the config.

        Args:
            cfg_dict: The config values.
            filename: The file the config was loaded from.
        """
        if cfg_dict is None:
            cfg_dict = {}
        elif not isinstance(cfg_dict, dict):
            msg = f"cfg_dict must be a dict, but got {type(cfg_dict)}"
            raise TypeError(msg)

        super().__setattr__("_cfg_dict", ConfigDict(cfg_dict))
        super().__setattr__("_filename", filename)

    @staticmethod
    def fromfile(filename: str) -> Config:
        """Load a config from a python file.

        Args:
            filename: Path to the python config file.

        Returns:
            The loaded config.
        """
        return Config(Config._file2dict(filename), filename=filename)

    @staticmethod
    def _file2dict(filename: str) -> dict:
        path = Path(filename).expanduser().resolve()
        if path.suffix != ".py":
            msg = f"Only python config files are supported, got: {path}"
            raise OSError(msg)

        cfg_dict = {
            name: value
            for name, value in runpy.run_path(str(path)).items()
            if not name.startswith("__")
        }

        if BASE_KEY in cfg_dict:
            base_files = cfg_dict.pop(BASE_KEY)
            if not isinstance(base_files, list):
                base_files = [base_files]

            base_cfg_dict: dict = {}
            for base_file in base_files:
                base_dict = Config._file2dict(str(path.parent / base_file))
                if base_cfg_dict.keys() & base_dict.keys():
                    msg = "Duplicate key is not allowed among bases"
                    raise KeyError(msg)
                base_cfg_dict.update(base_dict)

            cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)

        return cfg_dict

    @staticmethod
    def _merge_a_into_b(a: dict, b: dict) -> dict:
        """Merge dict `a` into a copy of dict `b`, values of `a` take precedence."""
        b = b.copy()
        for k, v in a.items():
            if isinstance(v, dict) and k in b and not v.pop(DELETE_KEY, False):
                if not isinstance(b[k], dict):
                    msg = (
                        f"{k}={v} in child config cannot inherit from base because "
                        f"{k} is a dict in the child config but is of type "
                        f"{type(b[k])} in base config. You may set `{DELETE_KEY}=True` "
                        "to ignore the base config"
                    )
                    raise TypeError(msg)
                b[k] = Config._merge_a_into_b(v, b[k])
            else:
                b[k] = v
        return b

    @property
    def filename(self) -> str | None:
        """Return the file the config was loaded from."""
        return self._filename

    def __repr__(self) -> str:
        """Return a string representation of the config."""
        return f"Config (path: {self.filename}): {self._cfg_dict!r}"

    def __len__(self) -> int:
        """Return the number of top-level config values."""
        return len(self._cfg_dict)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the top-level config keys."""
        return iter(self._cfg_dict)

    def __getattr__(self, name: str) -> Any:
        """Return a config value as attribute."""
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name: str) -> Any:
        """Return a config value as item."""
        return self._cfg_dict[name]

    def __setattr__(self, name: str, value: Any) -> None:
        """Set a config value as attribute."""
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name: str, value: Any) -> None:
        """Set a config value as item."""
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict[name] = value
