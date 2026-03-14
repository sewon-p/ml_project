"""Pipeline management dashboard v2 — run versioning + asset-based resume.

Usage:
    python scripts/dashboard.py
    # Open http://localhost:8501 in your browser
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Resolve project root (scripts/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

app = FastAPI(title="ML Pipeline Dashboard v2")

# Mount static files
static_dir = PROJECT_ROOT / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

RUNS_DIR = PROJECT_ROOT / "data" / "runs"

# Pipeline step definitions
PIPELINE_STEPS = {
    "generate": {
        "label": "Generate Scenarios",
        "cmd": [sys.executable, "-m", "scripts.generate_scenarios"],
    },
    "simulate": {
        "label": "SUMO Simulation",
        "cmd": [sys.executable, "-m", "scripts.run_simulation"],
    },
    "extract": {
        "label": "Feature Extraction",
        "cmd": [sys.executable, "-m", "scripts.extract_features"],
    },
    "residuals": {
        "label": "FD Residual Calc",
        "cmd": [sys.executable, "-m", "scripts.prepare_residuals"],
    },
    "train": {
        "label": "Model Training",
        "cmd": [sys.executable, "-m", "scripts.train"],
    },
    "evaluate": {
        "label": "Evaluation",
        "cmd": [sys.executable, "-m", "scripts.evaluate"],
    },
}

STAGE_ASSET_MAP = {
    "scenarios": {"next_steps": ["simulate", "extract", "residuals", "train", "evaluate"]},
    "fcd": {"next_steps": ["extract", "residuals", "train", "evaluate"]},
    "features": {"next_steps": ["residuals", "train", "evaluate"]},
    "model": {"next_steps": ["evaluate"]},
}


# ---------------------------------------------------------------------------
# RunManager
# ---------------------------------------------------------------------------

class RunManager:
    """Manage versioned pipeline runs under data/runs/."""

    def __init__(self, runs_dir: Path = RUNS_DIR) -> None:
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def list_runs(self) -> list[dict[str, Any]]:
        """List all runs (newest first), including legacy data/ if present."""
        runs: list[dict[str, Any]] = []

        # Scan data/runs/*
        if self.runs_dir.exists():
            for d in sorted(self.runs_dir.iterdir(), reverse=True):
                if d.is_dir():
                    info = self.scan_run(d)
                    if info:
                        runs.append(info)

        # Legacy run (data/scenarios.csv etc.)
        legacy = self._scan_legacy()
        if legacy:
            runs.append(legacy)

        return runs

    @staticmethod
    def _scan_assets(base_dir: Path, info: dict[str, Any],
                     model_dirs: list[Path] | None = None) -> bool:
        """Scan assets under base_dir and populate info['assets']. Returns True if any found."""
        import pandas as pd
        has_assets = False

        csv_path = base_dir / "scenarios.csv"
        if csv_path.exists():
            has_assets = True
            try:
                df = pd.read_csv(csv_path)
                info["assets"]["scenarios"] = {"count": len(df)}
            except Exception:
                info["assets"]["scenarios"] = {"count": "?"}

        fcd_dir = base_dir / "fcd"
        if fcd_dir.exists():
            fcd_dirs = [d for d in fcd_dir.iterdir() if d.is_dir()]
            if fcd_dirs:
                has_assets = True
                info["assets"]["fcd"] = {"count": len(fcd_dirs)}

        tab = base_dir / "features" / "dataset.parquet"
        if tab.exists():
            has_assets = True
            try:
                df = pd.read_parquet(tab)
                feat_info: dict[str, Any] = {
                    "samples": len(df),
                    "scenarios": (
                        int(df["scenario_id"].nunique())
                        if "scenario_id" in df.columns else 0
                    ),
                }
                # Scan available lanes / speed_limits for data filtering
                meta_path = base_dir / "features" / "metadata.parquet"
                meta_df = pd.read_parquet(meta_path) if meta_path.exists() else df
                if "num_lanes" in meta_df.columns:
                    feat_info["available_lanes"] = sorted(
                        int(v) for v in meta_df["num_lanes"].dropna().unique()
                    )
                if "speed_limit" in meta_df.columns:
                    # speed_limit is in m/s; convert to km/h (rounded int)
                    feat_info["available_speed_limits_kmh"] = sorted(
                        int(round(v * 3.6)) for v in meta_df["speed_limit"].dropna().unique()
                    )
                info["assets"]["features"] = feat_info
            except Exception:
                info["assets"]["features"] = {"samples": "?"}

        for mdir in (model_dirs or [base_dir / "outputs"]):
            if mdir.exists():
                models = list(mdir.glob("*.pt")) + list(mdir.glob("*.pkl"))
                if models:
                    has_assets = True
                    info["assets"]["model"] = {
                        "count": len(models),
                        "files": [m.name for m in models],
                    }

        return has_assets

    def scan_run(self, run_dir: Path) -> dict[str, Any] | None:
        """Scan a single run directory and return metadata."""
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            return None

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

        info: dict[str, Any] = {
            "run_id": run_dir.name,
            "dir": str(run_dir),
            "created": manifest.get("created", ""),
            "mode": manifest.get("mode", "unknown"),
            "source_run_id": manifest.get("source_run_id"),
            "completed_steps": manifest.get("completed_steps", []),
            "status": manifest.get("status", "unknown"),
            "num_scenarios": manifest.get("num_scenarios"),
            "scenario_info": manifest.get("scenario_info"),
            "fd_model": manifest.get("fd_model", "underwood"),
            "data_filters": manifest.get("data_filters"),
            "assets": {},
        }
        # For resume runs, also scan the source run's output directory
        extra_model_dirs: list[Path] | None = None
        source_run_id = manifest.get("source_run_id")
        if source_run_id:
            source_dir = self._resolve_run_dir(source_run_id)
            if source_dir:
                extra_model_dirs = [source_dir / "outputs"]
                # Also scan source features/fcd if this run has none
                if not (run_dir / "features").exists() and (source_dir / "features").exists():
                    self._scan_assets(source_dir, info, model_dirs=extra_model_dirs)
                    return info
        self._scan_assets(run_dir, info, model_dirs=extra_model_dirs)
        return info

    def _scan_legacy(self) -> dict[str, Any] | None:
        """Scan legacy flat data/ directory."""
        data_dir = PROJECT_ROOT / "data"
        info: dict[str, Any] = {
            "run_id": "legacy",
            "dir": str(data_dir),
            "created": "",
            "mode": "legacy",
            "source_run_id": None,
            "completed_steps": [],
            "status": "completed",
            "assets": {},
        }

        model_dirs = [PROJECT_ROOT / d for d in ("outputs", "outputs_xgboost")]
        has_assets = self._scan_assets(data_dir, info, model_dirs=model_dirs)

        if not has_assets:
            return None

        # Set created time from scenarios.csv
        csv_path = data_dir / "scenarios.csv"
        if csv_path.exists():
            info["created"] = datetime.fromtimestamp(
                csv_path.stat().st_mtime
            ).strftime("%Y-%m-%d %H:%M")

        # Infer completed steps
        if info["assets"].get("scenarios"):
            info["completed_steps"].append("generate")
        if info["assets"].get("fcd"):
            info["completed_steps"].append("simulate")
        if info["assets"].get("features"):
            info["completed_steps"].append("extract")

        return info

    def get_assets_for_stage(self, stage: str) -> list[dict[str, Any]]:
        """Get available assets for a given stage across all runs."""
        results: list[dict[str, Any]] = []

        for run_info in self.list_runs():
            if stage in run_info.get("assets", {}):
                asset = run_info["assets"][stage]
                results.append({
                    "run_id": run_info["run_id"],
                    "created": run_info["created"],
                    "dir": run_info["dir"],
                    **asset,
                })

        return results

    def create_run(self, mode: str, num_scenarios: int = 1000,
                   num_probes: int = 5, max_workers: int = 0,
                   device: str = "auto",
                   source_run_id: str | None = None,
                   source_stage: str | None = None,
                   steps: list[str] | None = None,
                   scenario_config: dict | None = None,
                   fd_model: str = "underwood",
                   data_filters: dict | None = None,
                   exclude_features: list[str] | None = None) -> tuple[str, Path]:
        """Create a new run directory with manifest and config overlay."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Build scenario info summary for run history display
        scenario_info = self._build_scenario_info(scenario_config)

        # Write manifest
        manifest: dict[str, Any] = {
            "run_id": run_id,
            "created": datetime.now().isoformat(),
            "mode": mode,
            "source_run_id": source_run_id,
            "source_stage": source_stage,
            "num_scenarios": num_scenarios,
            "num_probes": num_probes,
            "max_workers": max_workers,
            "fd_model": fd_model if steps and "residuals" in steps else None,
            "steps": steps or [],
            "completed_steps": [],
            "status": "pending",
            "scenario_info": scenario_info,
        }
        if data_filters:
            manifest["data_filters"] = data_filters
        if exclude_features:
            manifest["exclude_features"] = exclude_features
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Write per-run config overlay
        self._write_run_config(
            run_dir, source_run_id, source_stage, max_workers, device,
            scenario_config=scenario_config, fd_model=fd_model,
            data_filters=data_filters, exclude_features=exclude_features,
            steps=steps,
        )

        # Apply data filters: copy filtered feature files to new run dir
        if data_filters and source_run_id and source_stage == "features":
            self._apply_data_filters(run_dir, source_run_id, data_filters)

        return run_id, run_dir

    def _apply_data_filters(self, run_dir: Path, source_run_id: str,
                            data_filters: dict) -> None:
        """Filter source feature files by lanes/speed_limit and copy to run_dir/features/."""
        import numpy as np
        import pandas as pd

        source_dir = self._resolve_run_dir(source_run_id)
        if not source_dir:
            logger.warning("Cannot resolve source run %s for data filtering", source_run_id)
            return

        meta_path = source_dir / "features" / "metadata.parquet"
        tab_path = source_dir / "features" / "dataset.parquet"
        ts_path = source_dir / "features" / "timeseries.npz"

        if not meta_path.exists():
            logger.warning("metadata.parquet not found in %s, skipping filter", source_dir)
            return

        meta_df = pd.read_parquet(meta_path)
        total = len(meta_df)

        # Build boolean mask (OR within each filter, AND across filters)
        mask = pd.Series(True, index=meta_df.index)

        lanes = data_filters.get("lanes")
        if lanes and "num_lanes" in meta_df.columns:
            mask &= meta_df["num_lanes"].isin(lanes)

        speed_limits_kmh = data_filters.get("speed_limits_kmh")
        if speed_limits_kmh and "speed_limit" in meta_df.columns:
            # speed_limit is in m/s; compare with tolerance 0.1 m/s
            sl_ms = [v / 3.6 for v in speed_limits_kmh]
            sl_mask = pd.Series(False, index=meta_df.index)
            for target in sl_ms:
                sl_mask |= (meta_df["speed_limit"] - target).abs() < 0.1
            mask &= sl_mask

        kept = int(mask.sum())
        logger.info("Data filter: %d/%d samples kept", kept, total)

        # Write filtered files
        out_dir = run_dir / "features"
        out_dir.mkdir(parents=True, exist_ok=True)

        meta_df[mask].reset_index(drop=True).to_parquet(out_dir / "metadata.parquet")

        if tab_path.exists():
            tab_df = pd.read_parquet(tab_path)
            tab_df[mask].reset_index(drop=True).to_parquet(out_dir / "dataset.parquet")

        if ts_path.exists():
            ts_data = np.load(ts_path)
            idx = np.where(mask.values)[0]
            filtered = {key: ts_data[key][idx] for key in ts_data.files}
            np.savez(out_dir / "timeseries.npz", **filtered)

    def _write_run_config(self, run_dir: Path, source_run_id: str | None,
                          source_stage: str | None,
                          max_workers: int = 0, device: str = "auto",
                          scenario_config: dict | None = None,
                          fd_model: str = "underwood",
                          data_filters: dict | None = None,
                          exclude_features: list[str] | None = None,
                          steps: list[str] | None = None) -> None:
        """Generate run_config.yaml that inherits from default.yaml."""
        run_id = run_dir.name
        # Compute relative path from run_dir to configs/default.yaml
        rel_base = os.path.relpath(
            PROJECT_ROOT / "configs" / "default.yaml", run_dir
        ).replace("\\", "/")

        # Resolve device: "auto" means let the scripts decide
        resolved_device = device if device != "auto" else None

        overlay: dict[str, Any] = {
            "_base_": rel_base,
            "simulation": {
                "max_workers": max_workers if max_workers > 0 else "auto",
            },
            "output": {
                "scenarios_csv": f"data/runs/{run_id}/scenarios.csv",
                "fcd_dir": f"data/runs/{run_id}/fcd",
                "features_dir": f"data/runs/{run_id}/features",
            },
            "data": {
                "tabular_path": f"data/runs/{run_id}/features/dataset.parquet",
                "timeseries_path": f"data/runs/{run_id}/features/timeseries.npz",
                "metadata_path": f"data/runs/{run_id}/features/metadata.parquet",
            },
            "output_dir": f"data/runs/{run_id}/outputs",
        }

        if resolved_device:
            overlay["training"] = {"device": resolved_device}

        if exclude_features:
            overlay.setdefault("training", {})["exclude_features"] = exclude_features

        # FD residual correction: only enable when 'residuals' step is selected
        if steps and "residuals" in steps:
            overlay["residual_correction"] = {"enabled": True, "fd_model": fd_model}
        else:
            overlay["residual_correction"] = {"enabled": False}

        # If resuming from a source run, point the relevant source paths
        if source_run_id and source_stage:
            source_dir = self._resolve_run_dir(source_run_id)
            if source_dir:
                # Always use relative path from project root
                try:
                    src = str(source_dir.relative_to(PROJECT_ROOT)).replace("\\", "/")
                except ValueError:
                    src = str(source_dir).replace("\\", "/")
                # Map stage -> which paths to inherit from source
                if source_stage == "scenarios":
                    overlay["output"]["scenarios_csv"] = f"{src}/scenarios.csv"
                    overlay["simulation"]["scenarios_csv"] = f"{src}/scenarios.csv"
                elif source_stage == "fcd":
                    overlay["output"]["scenarios_csv"] = f"{src}/scenarios.csv"
                    overlay["simulation"]["scenarios_csv"] = f"{src}/scenarios.csv"
                    overlay["output"]["fcd_dir"] = f"{src}/fcd"
                elif source_stage == "features":
                    if data_filters:
                        # Filtered copy lives in run_dir/features/
                        local = f"data/runs/{run_id}/features"
                        overlay["data"]["tabular_path"] = f"{local}/dataset.parquet"
                        overlay["data"]["timeseries_path"] = f"{local}/timeseries.npz"
                        overlay["data"]["metadata_path"] = f"{local}/metadata.parquet"
                    else:
                        overlay["data"]["tabular_path"] = f"{src}/features/dataset.parquet"
                        overlay["data"]["timeseries_path"] = f"{src}/features/timeseries.npz"
                        overlay["data"]["metadata_path"] = f"{src}/features/metadata.parquet"
                elif source_stage == "model":
                    # Resolve data paths from source run's config (it may itself
                    # be a resume run without its own features/ directory).
                    data_paths = self._resolve_data_paths(source_dir)
                    overlay["data"]["tabular_path"] = data_paths["tabular_path"]
                    overlay["data"]["timeseries_path"] = data_paths["timeseries_path"]
                    overlay["data"]["metadata_path"] = data_paths["metadata_path"]
                    # Follow the chain to find the actual output_dir with models
                    overlay["output_dir"] = self._resolve_output_dir(source_dir)

        # Merge UI scenario config overrides into the simulation section
        if scenario_config:
            self._merge_scenario_config(overlay, scenario_config)

        config_path = run_dir / "run_config.yaml"
        config_path.write_text(
            yaml.dump(overlay, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )

    def _resolve_data_paths(self, run_dir: Path) -> dict[str, str]:
        """Resolve actual data paths from a run's config, following the chain.

        A resume run may not have its own features/ directory; its run_config
        points to the original source.  Read those paths instead of assuming
        ``{run_dir}/features/``.
        """
        try:
            src = str(run_dir.relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            src = str(run_dir).replace("\\", "/")
        defaults = {
            "tabular_path": f"{src}/features/dataset.parquet",
            "timeseries_path": f"{src}/features/timeseries.npz",
            "metadata_path": f"{src}/features/metadata.parquet",
        }
        cfg_path = run_dir / "run_config.yaml"
        if not cfg_path.exists():
            return defaults
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            data_cfg = cfg.get("data", {})
            return {
                "tabular_path": data_cfg.get("tabular_path", defaults["tabular_path"]),
                "timeseries_path": data_cfg.get("timeseries_path", defaults["timeseries_path"]),
                "metadata_path": data_cfg.get("metadata_path", defaults["metadata_path"]),
            }
        except Exception:
            return defaults

    def _resolve_output_dir(self, run_dir: Path) -> str:
        """Follow resume chain to find the output_dir that actually has models."""
        visited: set[str] = set()
        current = run_dir
        while current and str(current) not in visited:
            visited.add(str(current))
            out = current / "outputs"
            models = list(out.glob("*.pt")) + list(out.glob("*.pkl"))
            if models:
                try:
                    return str(out.relative_to(PROJECT_ROOT)).replace("\\", "/")
                except ValueError:
                    return str(out).replace("\\", "/")
            # Read this run's config to find its output_dir
            cfg_path = current / "run_config.yaml"
            if not cfg_path.exists():
                break
            try:
                with open(cfg_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                od = cfg.get("output_dir", "")
                if od and (PROJECT_ROOT / od).exists():
                    models = list((PROJECT_ROOT / od).glob("*.pt")) + list(
                        (PROJECT_ROOT / od).glob("*.pkl")
                    )
                    if models:
                        return od
                # Follow source_run_id chain via manifest
                manifest_path = current / "manifest.json"
                if manifest_path.exists():
                    m = json.loads(manifest_path.read_text(encoding="utf-8"))
                    src_id = m.get("source_run_id")
                    if src_id:
                        resolved = self._resolve_run_dir(src_id)
                        if resolved is not None:
                            current = resolved
                        continue
            except Exception:
                pass
            break
        # Fallback: use the run_dir itself
        try:
            return str((run_dir / "outputs").relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            return str(run_dir / "outputs").replace("\\", "/")

    def _resolve_run_dir(self, run_id: str) -> Path | None:
        """Resolve run_id to its directory."""
        if run_id == "legacy":
            return PROJECT_ROOT / "data"
        d = self.runs_dir / run_id
        return d if d.exists() else None

    @staticmethod
    def _merge_scenario_config(overlay: dict[str, Any], sc: dict) -> None:
        """Merge scenario_config dict from UI into overlay['simulation'].

        Keys like ``passenger_tau`` map to ``simulation.vehicle_types.passenger.tau``.
        Keys like ``per_lane_demand_vehph`` map to ``simulation.demand.per_lane_demand_vehph``.
        ``speed_limit_kmh`` maps to ``simulation.network.speed_limit_kmh``.
        ``num_lanes`` maps to ``simulation.network.num_lanes``.
        ``truck_ratio`` maps to ``simulation.vehicle_types.truck_ratio``.
        """
        sim = overlay.setdefault("simulation", {})
        net = sim.setdefault("network", {})
        demand = sim.setdefault("demand", {})
        vtypes = sim.setdefault("vehicle_types", {})

        for key, val in sc.items():
            if key == "speed_limit_kmh":
                net["speed_limit_kmh"] = val
            elif key == "num_lanes":
                net["num_lanes"] = val
            elif key == "per_lane_demand_vehph":
                demand["per_lane_demand_vehph"] = val
            elif key == "truck_ratio":
                vtypes["truck_ratio"] = val
            elif key.startswith("passenger_"):
                param = key[len("passenger_"):]
                vtypes.setdefault("passenger", {})[param] = val
            elif key.startswith("truck_"):
                param = key[len("truck_"):]
                vtypes.setdefault("truck", {})[param] = val

    def update_manifest(self, run_dir: Path, step: str | None = None,
                        status: str | None = None,
                        extra: dict[str, Any] | None = None) -> None:
        """Update manifest.json after step completion or status change."""
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            return

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if step and step not in manifest.get("completed_steps", []):
            manifest.setdefault("completed_steps", []).append(step)
        if status:
            manifest["status"] = status
        if extra:
            manifest.update(extra)
        manifest["updated"] = datetime.now().isoformat()

        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _build_scenario_info(self, scenario_config: dict | None) -> dict:
        """Build compact scenario summary for manifest display."""
        sc = scenario_config or {}

        # Read defaults from config file
        try:
            cfg_path = PROJECT_ROOT / "configs" / "default.yaml"
            with open(cfg_path, encoding="utf-8") as f:
                defaults = yaml.safe_load(f) or {}
            sim = defaults.get("simulation", {})
            net_d = sim.get("network", {})
            demand_d = sim.get("demand", {})
        except Exception:
            net_d, demand_d = {}, {}

        # Speed limits
        sl = sc.get("speed_limit_kmh") or net_d.get(
            "speed_limit_kmh", [50, 60, 80, 100]
        )

        # Num lanes
        nl = sc.get("num_lanes") or net_d.get("num_lanes", {"min": 1, "max": 3})
        if isinstance(nl, dict):
            nl_str = (
                str(nl["min"]) if nl.get("min") == nl.get("max")
                else f"{nl.get('min', 1)}\u2013{nl.get('max', 3)}"
            )
        else:
            nl_str = str(nl)

        # Demand per lane
        pld = sc.get("per_lane_demand_vehph")
        if pld and isinstance(pld, dict):
            d_min, d_max = pld.get("min", "?"), pld.get("max", "?")
        else:
            d_min = demand_d.get("per_lane_min_vehph", 200)
            d_max = demand_d.get("per_lane_max_vehph", 2200)
        demand_str = str(d_min) if d_min == d_max else f"{d_min}\u2013{d_max}"

        return {
            "speed_limits": sl,
            "lanes": nl_str,
            "demand_per_lane": demand_str,
        }

    def delete_run(self, run_id: str) -> bool:
        """Delete a run directory."""
        if run_id == "legacy":
            return False
        run_dir = self.runs_dir / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
            return True
        return False


run_manager = RunManager()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PipelineState:
    """Global mutable state for the running pipeline."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.running = False
        self.current_step = ""
        self.step_index = 0
        self.total_steps = 0
        self.log_lines: list[str] = []
        self.start_time: float = 0
        self.elapsed: float = 0
        self.process: asyncio.subprocess.Process | None = None
        self.error: str | None = None
        self.cancelled = False
        self.current_run_id: str | None = None
        self.current_run_dir: Path | None = None


state = PipelineState()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    steps: list[str]
    num_scenarios: int = 1000
    num_probes: int = 5
    max_workers: int = 0  # 0 = auto (cpu_count - 1)
    device: str = "auto"  # auto, cuda, mps, cpu
    model_types: list[str] = ["cnn1d"]  # cnn1d, lstm, xgboost, lightgbm
    mode: str = "from_scratch"  # "from_scratch" | "resume"
    source_run_id: str | None = None
    source_stage: str | None = None  # "scenarios" | "fcd" | "features" | "model"
    scenario_config: dict | None = None  # UI-supplied scenario parameter overrides
    fd_model: str = "underwood"  # greenshields | greenberg | underwood | drake | multi_regime
    data_filters: dict | None = None  # {"lanes": [2,3], "speed_limits_kmh": [80,100]}
    exclude_features: list[str] | None = None  # features to exclude from training/eval


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

_MODEL_CONFIG_MAP = {
    "cnn1d": "configs/models/cnn1d.yaml",
    "lstm": "configs/models/lstm.yaml",
    "xgboost": "configs/models/xgboost.yaml",
    "lightgbm": "configs/models/lightgbm.yaml",
}


def _write_model_config(base_config_path: str, out_path: str, model_type: str) -> None:
    """Write a config overlay that sets model.type and model.config for a specific training run."""
    # Use relative path from out_path dir to base_config_path
    rel = os.path.relpath(base_config_path, Path(out_path).parent).replace("\\", "/")
    model_overlay: dict[str, Any] = {"type": model_type}
    if model_type in _MODEL_CONFIG_MAP:
        model_overlay["config"] = _MODEL_CONFIG_MAP[model_type]
    overlay = {
        "_base_": rel,
        "model": model_overlay,
    }
    Path(out_path).write_text(
        yaml.dump(overlay, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )


async def _run_step(step_key: str, cmd: list[str],
                    extra_args: list[str] | None = None,
                    label: str | None = None) -> bool:
    """Run a single pipeline step as subprocess. Returns True on success."""
    full_cmd = cmd + (extra_args or [])
    step_label: str = label or str(PIPELINE_STEPS[step_key]["label"])
    state.current_step = step_label
    state.log_lines.append(f"\n>>> [{step_label}] Started")
    logger.info("Running: %s", " ".join(full_cmd))

    # CREATE_NEW_PROCESS_GROUP on Windows so taskkill /T can kill the tree
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    # PYTHONUNBUFFERED ensures worker output reaches us in real-time on Windows
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    proc = await asyncio.create_subprocess_exec(
        *full_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        env=env,
        **kwargs,
    )
    state.process = proc

    async for line_bytes in proc.stdout:  # type: ignore[union-attr]
        line = line_bytes.decode("utf-8", errors="replace").rstrip()
        state.log_lines.append(line)
        if len(state.log_lines) > 500:
            state.log_lines = state.log_lines[-500:]

    await proc.wait()
    success = proc.returncode == 0
    if success:
        state.log_lines.append(f"<<< [{state.current_step}] Done")
    else:
        state.error = f"{state.current_step} Failed (exit code {proc.returncode})"
        state.log_lines.append(f"<<< [{state.current_step}] Failed!")
    return success


async def _run_pipeline(req: RunRequest) -> None:
    """Execute pipeline with run versioning."""
    state.reset()
    state.running = True
    state.total_steps = len(req.steps)
    state.start_time = time.time()

    try:
        # Create run directory
        run_id, run_dir = run_manager.create_run(
            mode=req.mode,
            num_scenarios=req.num_scenarios,
            num_probes=req.num_probes,
            max_workers=req.max_workers,
            device=req.device,
            source_run_id=req.source_run_id,
            source_stage=req.source_stage,
            steps=req.steps,
            scenario_config=req.scenario_config,
            fd_model=req.fd_model,
            data_filters=req.data_filters,
            exclude_features=req.exclude_features,
        )
        state.current_run_id = run_id
        state.current_run_dir = run_dir
        config_path = str(run_dir / "run_config.yaml")

        run_manager.update_manifest(run_dir, status="running")
        state.log_lines.append(f"=== Run {run_id} Started (mode={req.mode}) ===")

        # Build effective step list: expand train/evaluate per model type
        # Auto-detect available models from output_dir if none specified
        model_types = list(req.model_types)
        if not model_types and "evaluate" in req.steps:
            with open(config_path, encoding="utf-8") as _f:
                run_cfg = yaml.safe_load(_f) or {}
            out_path = PROJECT_ROOT / run_cfg.get("output_dir", "")
            if out_path.exists():
                for f in out_path.iterdir():
                    if f.suffix == ".pt" and f.stem.endswith("_best"):
                        model_types.append(f.stem.replace("_best", ""))
                    elif f.suffix == ".pkl" and f.stem.endswith("_best"):
                        model_types.append(f.stem.replace("_best", ""))
                if model_types:
                    logger.info("Auto-detected models: %s", model_types)

        expanded_steps: list[tuple[str, str | None]] = []  # (step_key, model_type)
        for s in req.steps:
            if s in ("train", "evaluate") and model_types:
                for mt in model_types:
                    expanded_steps.append((s, mt))
            else:
                expanded_steps.append((s, None))

        state.total_steps = len(expanded_steps)

        for i, (step_key, model_type) in enumerate(expanded_steps):
            if state.cancelled:
                state.log_lines.append(">>> Pipeline cancelled")
                run_manager.update_manifest(run_dir, status="cancelled")
                break

            state.step_index = i + 1
            step = PIPELINE_STEPS[step_key]
            cmd: list[str] = list(step["cmd"]) + ["--config", config_path]
            extra_args: list[str] = []

            if step_key == "generate":
                extra_args = ["--num", str(req.num_scenarios)]
                extra_args += ["--output", str(run_dir / "scenarios.csv")]

            # For train/evaluate: use model-specific config overlay
            step_label: str | None = None
            if step_key in ("train", "evaluate") and model_type:
                model_cfg_path = run_dir / f"run_config_{model_type}.yaml"
                if not model_cfg_path.exists():
                    _write_model_config(config_path, str(model_cfg_path), model_type)
                cmd = list(step["cmd"]) + ["--config", str(model_cfg_path)]
                base_label = "Model Training" if step_key == "train" else "Evaluation"
                step_label = f"{base_label} ({model_type})"

            success = await _run_step(step_key, cmd, extra_args, label=step_label)
            if success:
                manifest_key = f"train:{model_type}" if model_type else step_key
                run_manager.update_manifest(run_dir, step=manifest_key)
            else:
                run_manager.update_manifest(run_dir, status="failed")
                break

        state.elapsed = time.time() - state.start_time
        if not state.error and not state.cancelled:
            run_manager.update_manifest(run_dir, status="completed",
                                        extra={"elapsed_seconds": round(state.elapsed, 1)})
            state.log_lines.append(f"\n=== Run {run_id} Done ({state.elapsed:.1f}s) ===")
    except Exception as e:
        state.error = str(e)
        state.log_lines.append(f"Error: {e}")
        if state.current_run_dir:
            run_manager.update_manifest(state.current_run_dir, status="failed")
    finally:
        state.running = False
        state.process = None


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = static_dir / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>dashboard.html not found</h1>")


@app.get("/api/status")
async def get_status():
    # Detect orphaned "running" state (process died externally)
    if state.running and state.process is not None:
        try:
            ret = state.process.returncode
            if ret is not None:
                # Process already exited but state wasn't cleaned up
                state.elapsed = time.time() - state.start_time if state.start_time else 0
                state.error = state.error or f"Process exited (exit code {ret})"
                state.running = False
                state.process = None
                if state.current_run_dir:
                    run_manager.update_manifest(state.current_run_dir, status="failed")
        except Exception:
            pass

    elapsed = time.time() - state.start_time if state.running else state.elapsed
    return {
        "running": state.running,
        "current_step": state.current_step,
        "step_index": state.step_index,
        "total_steps": state.total_steps,
        "elapsed": round(elapsed, 1),
        "error": state.error,
        "log_tail": state.log_lines[-50:] if state.log_lines else [],
        "run_id": state.current_run_id,
    }


@app.get("/api/runs")
async def get_runs():
    """List all runs."""
    return run_manager.list_runs()


@app.get("/api/assets-by-stage/{stage}")
async def get_assets_by_stage(stage: str):
    """Get available assets for a specific stage (for resume mode)."""
    if stage not in STAGE_ASSET_MAP:
        return {"error": f"Unknown stage: {stage}"}
    assets = run_manager.get_assets_for_stage(stage)
    next_steps = STAGE_ASSET_MAP[stage]["next_steps"]
    return {"stage": stage, "assets": assets, "next_steps": next_steps}


@app.get("/api/defaults/scenario")
async def get_scenario_defaults():
    """Return simulation parameter defaults from configs/default.yaml."""
    cfg_path = PROJECT_ROOT / "configs" / "default.yaml"
    if not cfg_path.exists():
        return {"error": "default.yaml not found"}
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        sim = cfg.get("simulation", {})
        net = sim.get("network", {})
        demand = sim.get("demand", {})
        vtypes = sim.get("vehicle_types", {})

        result: dict[str, Any] = {
            "speed_limit_kmh": net.get("speed_limit_kmh", [50, 60, 80, 100]),
            "num_lanes": net.get("num_lanes", {"min": 1, "max": 3}),
            "per_lane_demand_vehph": {
                "min": demand.get("per_lane_min_vehph", 200),
                "max": demand.get("per_lane_max_vehph", 2200),
            },
            "truck_ratio": vtypes.get("truck_ratio", {"min": 0.05, "max": 0.25}),
        }

        # Vehicle type params
        for vtype in ("passenger", "truck"):
            vt_cfg = vtypes.get(vtype, {})
            for param in ("tau", "decel", "minGap", "speedFactor", "sigma", "accel"):
                key = f"{vtype}_{param}"
                result[key] = vt_cfg.get(param, {})
            # Fixed length
            result[f"{vtype}_length"] = vt_cfg.get("length", 4.5 if vtype == "passenger" else 12.0)

        return result
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/run")
async def run_pipeline(req: RunRequest):
    if state.running:
        return {"error": "Pipeline is already running"}

    valid = [s for s in req.steps if s in PIPELINE_STEPS]
    if not valid:
        return {"error": "Please select steps to run"}

    req.steps = valid
    asyncio.create_task(_run_pipeline(req))
    return {"status": "started", "steps": valid, "mode": req.mode}


@app.post("/api/cancel")
async def cancel_pipeline():
    if not state.running:
        return {"status": "not_running"}
    state.cancelled = True
    _kill_process_tree()
    return {"status": "cancelling"}


@app.post("/api/force-reset")
async def force_reset():
    """Force-reset state when process died externally (Task Manager etc.)."""
    _kill_process_tree()
    state.log_lines.append(">>> Force reset")
    state.elapsed = time.time() - state.start_time if state.start_time else 0
    state.error = "Force terminated"
    state.running = False
    state.process = None
    if state.current_run_dir:
        run_manager.update_manifest(state.current_run_dir, status="cancelled")
    return {"status": "reset"}


def _kill_process_tree() -> None:
    """Kill the subprocess and all its children (multiprocessing workers)."""
    proc = state.process
    if proc is None:
        return
    pid = proc.pid
    try:
        if sys.platform == "win32":
            # taskkill /T kills the entire process tree on Windows
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, timeout=10,
            )
        else:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


@app.get("/api/logs")
async def stream_logs():
    """SSE endpoint for real-time log streaming."""
    async def event_generator():
        last_idx = 0
        while True:
            current_len = len(state.log_lines)
            if current_len > last_idx:
                for line in state.log_lines[last_idx:current_len]:
                    yield f"data: {line}\n\n"
                last_idx = current_len

            if not state.running and last_idx >= current_len:
                yield "data: [DONE]\n\n"
                break

            await asyncio.sleep(0.3)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/eval/{run_id}")
async def get_eval_results(run_id: str):
    """Get evaluation results (metrics + scatter data) for a run."""
    if run_id == "legacy":
        output_dirs = [PROJECT_ROOT / "outputs", PROJECT_ROOT / "outputs_xgboost"]
    else:
        run_dir = RUNS_DIR / run_id
        if not run_dir.exists():
            return {"error": f"Run {run_id} not found"}
        output_dirs = [run_dir / "outputs"]
        # Also check output_dir from run_config (may point to source run)
        cfg_path = run_dir / "run_config.yaml"
        if cfg_path.exists():
            try:
                with open(cfg_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                od = cfg.get("output_dir")
                if od:
                    resolved = PROJECT_ROOT / od
                    if resolved not in output_dirs:
                        output_dirs.append(resolved)
            except Exception:
                pass

    results = []
    for odir in output_dirs:
        if not odir.exists():
            continue
        for eval_json in odir.glob("eval_*.json"):
            try:
                data = json.loads(eval_json.read_text(encoding="utf-8"))
                results.append(data)
            except Exception:
                pass
    return {"run_id": run_id, "results": results}


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str):
    """Delete a run and its data."""
    if state.running and state.current_run_id == run_id:
        return {"error": "Cannot delete a running run"}
    if run_id == "legacy":
        return {"error": "Legacy data cannot be deleted via this API"}
    ok = run_manager.delete_run(run_id)
    if ok:
        return {"status": "deleted", "run_id": run_id}
    return {"error": f"Run {run_id} not found"}


@app.post("/api/clean")
async def clean_data():
    """Delete all generated data (legacy dirs)."""
    if state.running:
        return {"error": "Cannot clean while pipeline is running"}

    deleted = []
    dirs = [
        "data/fcd", "data/features", "data/sumo_networks",
        "outputs", "outputs_xgboost", "data/runs",
    ]
    for d in dirs:
        p = PROJECT_ROOT / d
        if p.exists():
            shutil.rmtree(p)
            deleted.append(d)
    csv = PROJECT_ROOT / "data" / "scenarios.csv"
    if csv.exists():
        csv.unlink()
        deleted.append("data/scenarios.csv")

    return {"status": "cleaned", "deleted": deleted}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8501, log_level="info")
