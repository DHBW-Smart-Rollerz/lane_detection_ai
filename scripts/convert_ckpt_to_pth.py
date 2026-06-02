import torch
import sys
import types

# 1. Create a dummy class to trick pickle
class DummyConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# 2. Create fake modules and inject them into Python's system modules
# This intercepts pickle's attempt to look for 'utils' and 'utils.config'
dummy_utils = types.ModuleType('utils')
dummy_config = types.ModuleType('config')

# Assign our dummy class to the expected name inside the fake module
dummy_config.Config = DummyConfig
dummy_config.ConfigDict = DummyConfig
dummy_utils.config = dummy_config

# Register the fake modules
sys.modules['utils'] = dummy_utils
sys.modules['utils.config'] = dummy_config
sys.modules['config'] = dummy_config

# ---------------------------------------------------------
# Now run your standard conversion code!
# ---------------------------------------------------------

ckpt_path = '/home/smartrollerz/smarty_workspace/src/lane_detection_ai/models/07.03.2026.ckpt'
pth_path = '/home/smartrollerz/smarty_workspace/src/lane_detection_ai/models/07.03.2026.pth'

# Load it (with weights_only=False so it accepts the dummy object)
checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)

# Extract just the weights
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# Save ONLY the weights as a clean PyTorch dictionary
torch.save(state_dict, pth_path)

print(f"Successfully extracted weights and saved to {pth_path}")