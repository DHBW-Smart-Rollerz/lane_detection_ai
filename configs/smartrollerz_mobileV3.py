dataset = "Smartrollerz"
data_root = "dataset/smartrollerz_bev"  # Need to be modified before running
epoch = 250 # 250
batch_size = 32
optimizer = "Adam"
learning_rate = 0.0006188572865938497 # 0.00625
weight_decay = 1.3115725431514786e-06
momentum = 0.9051601210183182
scheduler = "multi"
steps = [100, 150]
gamma = 0.1
warmup = "linear"
warmup_iters = 100
use_aux = False
griding_num = 200
backbone = "mobilenet-v3-small"
sim_loss_w = 0.0
shp_loss_w = 0.0
note = ""
log_path = "logs"
finetune = None
resume = None
test_model = ""
test_work_dir = ""
tta = True
num_lanes = 3
var_loss_power = 2.0
auto_backup = True
num_row = 55
num_col = 40
train_width = 512  # 128 # 256 #512  # 2048  # 1024
train_height = 384  # 96 # 192 #384  # 1536  # 768
num_cell_row = 100
num_cell_col = 100
original_image_width = 1364
original_image_height = 944
# train_width = 2048  # 128 # 256 #512  # 2048  # 1024
# train_height = 1536  # 96 # 192 #384  # 1536  # 768
# num_cell_row = 400
# num_cell_col = 400
mean_loss_w = 0.05
fc_norm = False
soft_loss = True
cls_loss_col_w = 1.0
cls_ext_col_w = 1.0
mean_loss_col_w = 0.05
eval_mode = "normal"
crop_ratio = 1.0
# Augmentation parameters
aug_translate_x = 50      # horizontally
aug_translate_y = 30      # vertically
aug_scale_min = 0.80
aug_scale_max = 1.20
aug_rotate_deg = 12.0

# Optimization target used by Lightning checkpoint callback
best_metric = "val/local_f1"
best_metric_mode = "max"

# Optuna defaults (can still be overridden via CLI)
optuna_n_trials = 30
optuna_timeout = None
optuna_study_name = "smartrollerz_lane_detection_optuna_study"
optuna_direction = "maximize"
optuna_sampler = "tpe"
optuna_seed = 42
optuna_pruner = "median"

optuna_space = {
    "learning_rate": {"type": "float", "low": 1e-4, "high": 2e-2, "log": True},
    "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
    "batch_size": {"type": "categorical", "choices": [16, 24, 32]},
    "optimizer": {"type": "categorical", "choices": ["SGD", "Adam"]},
    "momentum": {"type": "float", "low": 0.85, "high": 0.99},
}


# --- Optuna sampled hyperparameters for reproducibility ---
optuna_trial_number = 15
optuna_sampled_params = {'learning_rate': 0.00037924161599401605, 'weight_decay': 3.0864677182563864e-05, 'batch_size': 32, 'optimizer': 'Adam', 'momentum': 0.9822902548759617}
learning_rate = 0.00037924161599401605
weight_decay = 3.0864677182563864e-05
batch_size = 32
optimizer = 'Adam'
momentum = 0.9822902548759617

dataset = "Smartrollerz"
# data_root = "dataset/smartrollerz_bev"  # Need to be modified before running
# epoch = 250 # 250
# batch_size = 32
# optimizer = "SGD"
# learning_rate = 0.008 # 0.00625
# weight_decay = 0.0001
# momentum = 0.9
# scheduler = "multi"
# steps = [100, 150]
# gamma = 0.1
# warmup = "linear"
# warmup_iters = 100
# use_aux = False
# griding_num = 200
# backbone = "18"
# sim_loss_w = 0.0
# shp_loss_w = 0.0
# note = ""
# log_path = "logs"
# finetune = None
# resume = None
# test_model = ""
# test_work_dir = ""
# tta = True
# num_lanes = 3
# var_loss_power = 2.0
# auto_backup = True
# num_row = 55
# num_col = 40
# train_width = 512  # 128 # 256 #512  # 2048  # 1024
# train_height = 384  # 96 # 192 #384  # 1536  # 768
# num_cell_row = 100
# num_cell_col = 100
# mean_loss_w = 0.05
# fc_norm = False
# soft_loss = True
# cls_loss_col_w = 1.0
# cls_ext_col_w = 1.0
# mean_loss_col_w = 0.05
# eval_mode = "normal"
# crop_ratio = 1.0
# # Augmentation parameters
# aug_translate_x = 50      # horizontally
# aug_translate_y = 30      # vertically
