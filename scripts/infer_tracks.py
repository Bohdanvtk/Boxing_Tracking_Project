from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


from boxing_project.tracking.infer_runner import InferRunner


def main():
    InferRunner(ROOT / "configs" / "infer_tracks.yaml").run()


if __name__ == "__main__":
    main()
