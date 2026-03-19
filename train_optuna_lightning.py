import copy
import json
import os
import re
import subprocess
import sys
import random
import time
from datetime import datetime
from typing import Any, Dict, Optional
import warnings
from urllib.parse import urlparse

import optuna
import pytorch_lightning as pl
import torch
import mlflow
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from lightning.datamodule import SmartrollerzDataModule
from lightning.module import LaneDetectionLightningModule
from utils.common import get_work_dir, merge_config


def _configure_runtime_cache_dirs(base_dir: str) -> None:
    cache_root = os.path.join(base_dir, ".cache")
    config_root = os.path.join(base_dir, ".config")
    torch_home = os.path.join(cache_root, "torch")
    mpl_config = os.path.join(config_root, "matplotlib")

    os.environ.setdefault("XDG_CACHE_HOME", cache_root)
    os.environ.setdefault("TORCH_HOME", torch_home)
    os.environ.setdefault("MPLCONFIGDIR", mpl_config)

    os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
    os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


def _parse_gpu_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return max(0, value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
    return 0


def _resolve_parallel_jobs(cfg) -> int:
    cfg_jobs = getattr(cfg, "optuna_parallel_jobs", None)
    if cfg_jobs is not None:
        try:
            cfg_jobs_int = int(cfg_jobs)
            if cfg_jobs_int > 0:
                return cfg_jobs_int
        except (TypeError, ValueError):
            pass

    slurm_jobs = _parse_gpu_count(os.environ.get("SLURM_GPUS_ON_NODE"))
    if slurm_jobs > 0:
        return slurm_jobs

    cuda_jobs = torch.cuda.device_count()
    return max(1, int(cuda_jobs) if cuda_jobs else 1)


def _is_optuna_worker_process() -> bool:
    return os.environ.get("OPTUNA_WORKER_MODE", "0") == "1"


def _get_optuna_worker_id() -> int:
    raw = os.environ.get("OPTUNA_WORKER_ID", "0")
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _launch_optuna_workers(worker_count: int) -> None:
    cmd = [sys.executable, os.path.abspath(__file__), *sys.argv[1:]]
    procs = []
    for gpu_idx in range(worker_count):
        env = os.environ.copy()
        env["OPTUNA_WORKER_MODE"] = "1"
        env["OPTUNA_WORKER_ID"] = str(gpu_idx)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        proc = subprocess.Popen(cmd, env=env)
        procs.append(proc)

    failed = []
    for idx, proc in enumerate(procs):
        rc = proc.wait()
        if rc != 0:
            failed.append((idx, rc))

    if failed:
        details = ", ".join([f"worker{idx}:rc={rc}" for idx, rc in failed])
        raise RuntimeError(f"Optuna worker processes failed: {details}")


def _write_study_summary(study_root: str, study: optuna.Study) -> str:
    summary_path = os.path.join(study_root, "best_trial.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "study_name": study.study_name,
                "best_value": study.best_value,
                "best_trial_number": study.best_trial.number,
                "best_params": study.best_trial.params,
            },
            f,
            indent=2,
        )
    return summary_path


def _sample_params(trial: optuna.Trial, cfg) -> Dict[str, Any]:
    space = getattr(cfg, "optuna_space", None)
    if not isinstance(space, dict) or len(space) == 0:
        space = {
            "learning_rate": {"type": "float", "low": 1e-4, "high": 2e-2, "log": True},
            "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
            "batch_size": {"type": "categorical", "choices": [16, 24, 32]},
            "optimizer": {"type": "categorical", "choices": ["SGD", "Adam"]},
            "momentum": {"type": "float", "low": 0.85, "high": 0.99},
            "fc_norm": {"type": "categorical", "choices": [False, True]},
        }

    sampled: Dict[str, Any] = {}
    for name, spec in space.items():
        t = spec.get("type")
        if t == "float":
            sampled[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False))
            )
        elif t == "int":
            sampled[name] = trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1))
            )
        elif t == "categorical":
            sampled[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ValueError(f"Unsupported optuna_space type for '{name}': {t}")
    return sampled


def _write_trial_config(
    base_config_path: str,
    trial_dir: str,
    sampled: Dict[str, Any],
    trial_number: int,
    extra_overrides: Optional[Dict[str, Any]] = None,
) -> str:
    os.makedirs(trial_dir, exist_ok=True)
    dst = os.path.join(trial_dir, "config.py")
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_text = f.read()
    with open(dst, "w", encoding="utf-8") as f:
        f.write(base_text)
        f.write("\n\n# --- Optuna sampled hyperparameters for reproducibility ---\n")
        f.write(f"optuna_trial_number = {trial_number}\n")
        f.write(f"optuna_sampled_params = {repr(sampled)}\n")
        for k, v in sampled.items():
            f.write(f"{k} = {repr(v)}\n")
        if extra_overrides:
            f.write("\n# --- Extra runtime overrides ---\n")
            for k, v in extra_overrides.items():
                f.write(f"{k} = {repr(v)}\n")
    return dst


def _flush_and_close_loggers(loggers):
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


def _extract_state_dict_from_checkpoint(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        if all(torch.is_tensor(v) for v in ckpt_obj.values()):
            return ckpt_obj
    raise ValueError("Unsupported checkpoint format: expected {'model': ...}, {'state_dict': ...}, or raw state_dict.")


def _normalize_prefix_module(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            normalized[k[len("module."):]] = v
        else:
            normalized[k] = v
    return normalized


def _normalize_prefix_model(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            normalized[k[len("model."):]] = v
        else:
            normalized[k] = v
    return normalized


def _best_key_alignment(
    source_state: Dict[str, torch.Tensor], target_state: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    target_keys = set(target_state.keys())
    candidates = [
        source_state,
        _normalize_prefix_module(source_state),
        _normalize_prefix_model(source_state),
        _normalize_prefix_model(_normalize_prefix_module(source_state)),
    ]

    best_state = candidates[0]
    best_overlap = -1
    for candidate in candidates:
        overlap = sum(1 for k in candidate.keys() if k in target_keys)
        if overlap > best_overlap:
            best_overlap = overlap
            best_state = candidate
    return best_state


def _select_finetune_subset(
    state_dict: Dict[str, torch.Tensor],
    backbone_only: bool,
) -> Dict[str, torch.Tensor]:
    if not backbone_only:
        return state_dict
    # parsingNet backbone lives under `model.*`; head layers (`cls.*`, `pool.*`) are excluded.
    return {k: v for k, v in state_dict.items() if k.startswith("model.")}


def _load_finetune_weights_into_module(
    pl_module: pl.LightningModule,
    finetune_path: str,
    backbone_only: bool = True,
    strict: bool = False,
) -> Dict[str, int]:
    if not os.path.isfile(finetune_path):
        raise FileNotFoundError(f"Finetune checkpoint not found: {finetune_path}")

    checkpoint = torch.load(finetune_path, map_location="cpu")
    raw_state = _extract_state_dict_from_checkpoint(checkpoint)

    model_obj = getattr(pl_module, "model", None)
    if model_obj is None:
        raise RuntimeError("Lightning module has no `model` attribute for finetuning.")

    aligned_state = _best_key_alignment(raw_state, model_obj.state_dict())
    filtered_state = _select_finetune_subset(aligned_state, backbone_only=backbone_only)
    if len(filtered_state) == 0:
        raise RuntimeError(
            "No finetune weights selected for loading. "
            "If checkpoint uses full-model keys, set finetune_backbone_only=False."
        )

    incompatible = model_obj.load_state_dict(filtered_state, strict=strict)
    loaded_keys = len(filtered_state)
    missing_keys = len(incompatible.missing_keys)
    unexpected_keys = len(incompatible.unexpected_keys)
    print(
        "[Finetune] Loaded",
        loaded_keys,
        "keys from",
        finetune_path,
        f"(backbone_only={backbone_only}, strict={strict}, missing={missing_keys}, unexpected={unexpected_keys})",
    )
    return {
        "loaded_keys": loaded_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }


class BestWeightsSaver(Callback):
    def __init__(self, monitor: str, mode: str, save_path: str):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.save_path = save_path
        self.best_score: Optional[float] = None

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score
        return score > self.best_score

    def on_validation_end(self, trainer, pl_module) -> None:
        metric = trainer.callback_metrics.get(self.monitor)
        score = self._to_float(metric)
        if score is None:
            return

        if self._is_better(score):
            self.best_score = score
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(pl_module.state_dict(), self.save_path)


class BestMetricTracker(Callback):
    def __init__(self, monitor: str, mode: str = "max"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_score: Optional[float] = None

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score
        return score > self.best_score

    def on_validation_end(self, trainer, pl_module) -> None:
        metric = trainer.callback_metrics.get(self.monitor)
        score = self._to_float(metric)
        if score is None:
            return
        if self._is_better(score):
            self.best_score = score


def _print_used_config(args, cfg) -> None:
    print(f"[Config] Using config file: {os.path.abspath(args.config)}")
    print("[Config] Effective merged config:")
    try:
        print(cfg.pretty_text)
    except Exception:
        print(str(cfg))


def _is_optuna_alembic_race_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "UNIQUE constraint failed: alembic_version.version_num" in msg
        or "alembic_version.version_num" in msg
    )


def _create_or_load_study_with_retry(
    *,
    study_name: str,
    storage: str,
    direction: str,
    sampler,
    pruner,
    max_retries: int = 8,
    base_sleep_s: float = 0.2,
):
    last_exc = None
    for attempt in range(max_retries):
        try:
            return optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction=direction,
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True,
            )
        except Exception as exc:
            if not _is_optuna_alembic_race_error(exc):
                raise
            last_exc = exc
            sleep_s = base_sleep_s * (2 ** attempt) + random.uniform(0.0, 0.25)
            print(
                f"[Optuna][WARN] create_study race detected (attempt={attempt + 1}/{max_retries}); "
                f"retrying in {sleep_s:.2f}s"
            )
            time.sleep(sleep_s)

    # Fallback: if schema is already initialized by another worker, load study directly.
    try:
        return optuna.load_study(study_name=study_name, storage=storage)
    except Exception:
        if last_exc is not None:
            raise last_exc
        raise


def _safe_mlflow_log_artifact(mlflow_logger, run_id: str, local_path: str, artifact_path: str | None = None) -> bool:
    """
    Logs artifact if MLflow tracking/artifact URI combination is valid.
    Returns True if logged, False if skipped/failed (without raising).
    """
    try:
        tracking_uri = mlflow.get_tracking_uri()
        parsed_tracking = urlparse(tracking_uri or "")
        tracking_scheme = (parsed_tracking.scheme or "").lower()

        run_info = mlflow_logger.experiment.get_run(run_id).info
        run_artifact_uri = (run_info.artifact_uri or "").lower()

        # mlflow-artifacts URI requires HTTP(S) tracking URI
        if run_artifact_uri.startswith("mlflow-artifacts:") and tracking_scheme not in {"http", "https"}:
            warnings.warn(
                f"Skipping artifact logging for run_id={run_id}: "
                f"artifact_uri='{run_info.artifact_uri}' requires HTTP(S) tracking URI, "
                f"but tracking_uri='{tracking_uri}'."
            )
            return False

        mlflow_logger.experiment.log_artifact(
            run_id=run_id,
            local_path=local_path,
            artifact_path=artifact_path,
        )
        return True
    except Exception as exc:
        warnings.warn(f"Artifact logging failed (non-fatal): {exc}")
        return False


def main():
    torch.backends.cudnn.benchmark = True
    args, cfg = merge_config()
    _print_used_config(args, cfg)

    runtime_base = getattr(cfg, "log_path", None) or os.getcwd()
    _configure_runtime_cache_dirs(runtime_base)

    n_trials = int(getattr(cfg, "optuna_n_trials", 20) or 20)
    timeout = getattr(cfg, "optuna_timeout", None)
    direction = getattr(cfg, "optuna_direction", "maximize")
    seed = int(getattr(cfg, "optuna_seed", 42) or 42)
    sampler_name = getattr(cfg, "optuna_sampler", "tpe")
    pruner_name = getattr(cfg, "optuna_pruner", "median")
    study_name = getattr(cfg, "optuna_study_name", "smartrollerz_lane_detection_optuna")
    storage = getattr(cfg, "optuna_storage", None)
    study_root = os.path.join(getattr(cfg, "log_path", os.getcwd()), "optuna", study_name)
    os.makedirs(study_root, exist_ok=True)
    parallel_jobs = max(1, _resolve_parallel_jobs(cfg))
    worker_id = _get_optuna_worker_id()

    if parallel_jobs > 1 and storage and not _is_optuna_worker_process():
        print(f"[Optuna] Launching {parallel_jobs} parallel workers (1 GPU per worker).")
        _launch_optuna_workers(parallel_jobs)

        study = optuna.load_study(study_name=study_name, storage=storage)
        summary_path = _write_study_summary(study_root, study)
        print(f"[Optuna] Best value: {study.best_value}")
        print(f"[Optuna] Best trial: {study.best_trial.number}")
        print(f"[Optuna] Best params: {study.best_trial.params}")
        print(f"[Optuna] Summary written to: {summary_path}")
        return

    if parallel_jobs > 1 and not storage and not _is_optuna_worker_process():
        print("[Optuna] Parallel mode requested but no --optuna_storage was provided. Falling back to single worker.")
        parallel_jobs = 1

    sampler_seed = seed + (worker_id * 1000003)
    if sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=sampler_seed)
    else:
        sampler = optuna.samplers.TPESampler(
            seed=sampler_seed,
            multivariate=True,
            constant_liar=(parallel_jobs > 1),
        )

    if pruner_name == "none":
        pruner = optuna.pruners.NopPruner()
    else:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    # Replace direct create_study(...) with retry-safe helper.
    study = _create_or_load_study_with_retry(
        study_name=cfg.optuna_study_name,
        storage=cfg.optuna_storage,
        direction=cfg.optuna_direction,
        sampler=sampler,
        pruner=pruner,
    )

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = copy.deepcopy(cfg)
        sampled = _sample_params(trial, trial_cfg)
        for k, v in sampled.items():
            setattr(trial_cfg, k, v)

        finetune_path = getattr(trial_cfg, "finetune", None)
        finetune_backbone_only = bool(getattr(trial_cfg, "finetune_backbone_only", True))
        finetune_strict = bool(getattr(trial_cfg, "finetune_strict", False))

        # Disable automatic module-side finetune loading so this script controls
        # checkpoint format handling and logging consistently.
        if finetune_path:
            setattr(trial_cfg, "finetune_active", True)
            setattr(trial_cfg, "finetune", None)

        pl.seed_everything(seed + trial.number, workers=True)

        trial_dir = os.path.join(study_root, f"trial_{trial.number:04d}")
        os.makedirs(trial_dir, exist_ok=True)

        datamodule = SmartrollerzDataModule(trial_cfg)
        datamodule.setup("fit")
        datamodule.setup("validate")
        if not datamodule.has_validation_loader:
            raise RuntimeError("Validation loader is required for Optuna objective metric.")

        model = LaneDetectionLightningModule(trial_cfg)
        if datamodule.train_loader_len is None:
            raise RuntimeError("DataModule failed to initialize train loader length.")
        model.set_iters_per_epoch(datamodule.train_loader_len)

        finetune_stats: Optional[Dict[str, int]] = None
        if finetune_path:
            finetune_stats = _load_finetune_weights_into_module(
                model,
                finetune_path=finetune_path,
                backbone_only=finetune_backbone_only,
                strict=finetune_strict,
            )

        monitor_metric = getattr(trial_cfg, "best_metric", "val/local_f1")
        monitor_mode = getattr(trial_cfg, "best_metric_mode", "max")

        best_weights_path = os.path.join(trial_dir, "best_weights.pth")
        best_ckpt_cb = ModelCheckpoint(
            dirpath=trial_dir,
            filename="best-{epoch}",
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=1,
            save_last=False,
            auto_insert_metric_name=True,
        )
        best_weights_cb = BestWeightsSaver(
            monitor=monitor_metric,
            mode=monitor_mode,
            save_path=best_weights_path,
        )
        best_f1_cb = BestMetricTracker(monitor="val/local_f1", mode="max")
        lr_monitor = LearningRateMonitor(logging_interval="step")

        tb_logger = TensorBoardLogger(save_dir=trial_dir, name="events", default_hp_metric=False)
        session_timestamp = os.environ.get("OPTUNA_SESSION_TS")
        if not session_timestamp:
            session_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            os.environ["OPTUNA_SESSION_TS"] = session_timestamp

        mlflow_logger = MLFlowLogger(
            experiment_name=getattr(trial_cfg, "mlflow_experiment", "smartrollerz_lane_detection"),
            tracking_uri=getattr(trial_cfg, "mlflow_uri", None),
            run_name=f"optuna-{session_timestamp}-trial-{trial.number:04d}",
        )
        mlflow_hparams = dict(sampled)
        mlflow_hparams["finetune_enabled"] = bool(finetune_path)
        if finetune_path:
            mlflow_hparams["finetune_path"] = finetune_path
            mlflow_hparams["finetune_backbone_only"] = finetune_backbone_only
            mlflow_hparams["finetune_strict"] = finetune_strict
        mlflow_logger.log_hyperparams(mlflow_hparams)

        extra_overrides: Dict[str, Any] = {}
        if finetune_path:
            extra_overrides["finetune"] = finetune_path
            extra_overrides["finetune_backbone_only"] = finetune_backbone_only
            extra_overrides["finetune_strict"] = finetune_strict
        config_artifact = _write_trial_config(
            os.path.abspath(args.config),
            trial_dir,
            sampled,
            trial.number,
            extra_overrides=extra_overrides if extra_overrides else None,
        )
        # Replace direct call:
        # mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, config_artifact, artifact_path="config")
        _safe_mlflow_log_artifact(
            mlflow_logger=mlflow_logger,
            run_id=mlflow_logger.run_id,
            local_path=config_artifact,
            artifact_path="config",
        )

        devices = 1 if torch.cuda.is_available() else "auto"

        trainer = pl.Trainer(
            max_epochs=trial_cfg.epoch,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=devices,
            strategy="auto",
            enable_checkpointing=True,
            reload_dataloaders_every_n_epochs=1,
            default_root_dir=trial_dir,
            logger=[tb_logger, mlflow_logger],
            callbacks=[best_ckpt_cb, best_weights_cb, best_f1_cb, lr_monitor],
            log_every_n_steps=getattr(trial_cfg, "log_interval", 20),
            gradient_clip_val=getattr(trial_cfg, "grad_clip", 0.0) or None,
            deterministic=getattr(trial_cfg, "deterministic", False),
            limit_val_batches=1.0,
            num_sanity_val_steps=0,
            sync_batchnorm=(torch.cuda.is_available() and torch.cuda.device_count() > 1),
        )

        trainer.fit(model, datamodule=datamodule)

        best_score = best_weights_cb.best_score
        if best_score is None:
            value = trainer.callback_metrics.get(monitor_metric)
            if value is None:
                raise RuntimeError(f"Could not read objective metric '{monitor_metric}'.")
            best_score = float(value.detach().cpu().item() if torch.is_tensor(value) else value)

        best_f1 = best_f1_cb.best_score
        if best_f1 is None:
            f1_value = trainer.callback_metrics.get("val/local_f1")
            if f1_value is not None:
                best_f1 = float(f1_value.detach().cpu().item() if torch.is_tensor(f1_value) else f1_value)
        if best_f1 is not None:
            mlflow_logger.experiment.log_param(mlflow_logger.run_id, "best_f1", float(best_f1))
        if finetune_stats is not None:
            mlflow_logger.experiment.log_param(
                mlflow_logger.run_id,
                "finetune_loaded_keys",
                int(finetune_stats["loaded_keys"]),
            )
            mlflow_logger.experiment.log_param(
                mlflow_logger.run_id,
                "finetune_missing_keys",
                int(finetune_stats["missing_keys"]),
            )
            mlflow_logger.experiment.log_param(
                mlflow_logger.run_id,
                "finetune_unexpected_keys",
                int(finetune_stats["unexpected_keys"]),
            )

        if os.path.exists(best_weights_path):
            _safe_mlflow_log_artifact(
                mlflow_logger=mlflow_logger,
                run_id=mlflow_logger.run_id,
                local_path=best_weights_path,
                artifact_path="weights",
            )
        if best_ckpt_cb.best_model_path and os.path.exists(best_ckpt_cb.best_model_path):
            _safe_mlflow_log_artifact(
                mlflow_logger=mlflow_logger,
                run_id=mlflow_logger.run_id,
                local_path=best_ckpt_cb.best_model_path,
                artifact_path="checkpoints",
            )

        score = float(best_score)
        trial.report(score, step=trial_cfg.epoch)
        _flush_and_close_loggers([tb_logger, mlflow_logger])
        return score

    callbacks = [MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL))]

    study.optimize(
        objective,
        n_trials=n_trials if parallel_jobs == 1 else None,
        timeout=timeout,
        gc_after_trial=True,
        show_progress_bar=(parallel_jobs == 1 and not _is_optuna_worker_process()),
        catch=(OSError, RuntimeError),
        callbacks=callbacks,
    )

    summary_path = _write_study_summary(study_root, study)

    print(f"[Optuna] Best value: {study.best_value}")
    print(f"[Optuna] Best trial: {study.best_trial.number}")
    print(f"[Optuna] Best params: {study.best_trial.params}")
    print(f"[Optuna] Summary written to: {summary_path}")


if __name__ == "__main__":
    main()