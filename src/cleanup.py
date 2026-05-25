import shutil

from src.logging_utils import log_step


def remove_path(path):
    if path is None or not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    log_step(f"removed intermediate: {path}")


def cleanup_paths(paths):
    for path in paths:
        remove_path(path)
