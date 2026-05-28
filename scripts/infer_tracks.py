import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from pathlib import Path
from boxing_project.tracking.infer_runner import InferRunner

ROOT = Path(__file__).resolve().parents[1]


def main():
    InferRunner(ROOT / "configs" / "infer_tracks.yaml").run()


if __name__ == "__main__":
    main()