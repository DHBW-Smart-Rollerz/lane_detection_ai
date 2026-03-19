dataset = "Smartrollerz"
data_root = "dataset/smartrollerz"  # Need to be modified before running
epoch = 120
batch_size = 32
optimizer = "Adam"
learning_rate = 0.0003
weight_decay = 1.0e-05
momentum = 0.9027391247287574
scheduler = "multi"
steps = [60, 90]
gamma = 0.1
warmup = "linear"
warmup_iters = 50
use_aux = False
griding_num = 200
backbone = "18"
sim_loss_w = 0.0
shp_loss_w = 0.0
note = "_finetune_tusimple_res18"
log_path = "logs"
finetune = "pretrained_models/tusimple_res18.pth"  # Path to pretrained TuSimple model
finetune_backbone_only = True  # Load only backbone weights for domain adaptation
finetune_strict = False  # Allow head mismatch between TuSimple and Smartrollerz

# Optional stage-wise finetune behavior (applies only when `finetune` is set)
finetune_stagewise = True
finetune_head_warmup_epochs = 6  # train new head only first, then unfreeze backbone

# Optional discriminative LR (applies only when `finetune` is set)
finetune_discriminative_lr = True
finetune_backbone_lr_scale = 0.2  # backbone lr = learning_rate * this scale
finetune_head_lr_scale = 1.0  # head lr = learning_rate * this scale
finetune_backbone_prefix = "model."  # parsingNet backbone parameter prefix

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
num_cell_row = 250
num_cell_col = 250
original_image_width = 2064
original_image_height = 1544
mean_loss_w = 0.05
fc_norm = False
soft_loss = True
cls_loss_col_w = 1.0
cls_ext_col_w = 1.0
mean_loss_col_w = 0.05
eval_mode = "normal"
crop_ratio = 1.0
# Augmentation parameters
aug_translate_x = 20      # horizontally
aug_translate_y = 12      # vertically
aug_scale_min = 0.90
aug_scale_max = 1.10
aug_rotate_deg = 6.0


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
    # Finetuning search space: lower LR/regularization than full training.
    "learning_rate": {"type": "float", "low": 5e-5, "high": 2e-3, "log": True},
    "weight_decay": {"type": "float", "low": 1e-7, "high": 1e-4, "log": True},
    "batch_size": {"type": "categorical", "choices": [16, 24, 32]},
    # "optimizer": {"type": "categorical", "choices": ["Adam", "SGD"]},
    "momentum": {"type": "float", "low": 0.88, "high": 0.97},
    "finetune_head_warmup_epochs": {"type": "int", "low": 3, "high": 15, "step": 1},
    "finetune_backbone_lr_scale": {"type": "float", "low": 0.2, "high": 1.0, "log": True},
}
