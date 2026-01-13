from pathlib import Path
from typing import Dict, Optional


def resolve_dataset_path(dataset_cfg: Dict, filename_key: Optional[str] = None) -> Path:
    if "path" not in dataset_cfg:
        raise KeyError("DATASETS entry must define 'path'")

    base_path = Path(dataset_cfg["path"])
    if filename_key is None:
        return base_path

    elif filename_key not in dataset_cfg:
        raise KeyError(f"Missing '{filename_key}' in DATASETS config")

    return base_path / dataset_cfg[filename_key]
