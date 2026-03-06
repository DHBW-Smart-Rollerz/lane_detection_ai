"""
Configuration class for managing and merging settings from different sources.
"""
from pathlib import Path
import os
import numpy as np
import warnings
import copy


class Config:
    def __init__(self, cfg_dict=None, **kwargs):
        if cfg_dict is not None:
            self._cfg_dict = cfg_dict
        else:
            self._cfg_dict = {}
        
        # Update with any keyword arguments
        for k, v in kwargs.items():
            self._cfg_dict[k] = v

    def __getitem__(self, item):
        return self._cfg_dict[item]

    def __setitem__(self, key, value):
        self._cfg_dict[key] = value

    def __getattr__(self, name):
        # Prevent recursion by checking for _cfg_dict existence first
        if name == '_cfg_dict':
            raise AttributeError(f"Config object has no attribute '{name}'")
        
        # Safely get _cfg_dict to avoid recursion during unpickling
        cfg_dict = object.__getattribute__(self, '_cfg_dict')
        
        # If the attribute is not in _cfg_dict, raise AttributeError
        if name not in cfg_dict:
            raise AttributeError(f"Config object has no attribute '{name}'")
        
        return cfg_dict[name]

    def __repr__(self):
        return f"Config({self._cfg_dict})"

    def merge_from_file(self, filename):
        """Merge config from a file."""
        # Implementation for merging from a file
        pass

    def merge_from_list(self, cfg_list):
        """Merge config from a list."""
        # Implementation for merging from a list
        pass

    def to_dict(self):
        """Convert the config to a dictionary."""
        return copy.deepcopy(self._cfg_dict)

    def update(self, other_cfg):
        """Update the config with another config object."""
        if not isinstance(other_cfg, Config):
            raise ValueError("Can only update from another Config object")
        
        self._cfg_dict.update(other_cfg.to_dict())
