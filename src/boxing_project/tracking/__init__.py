from pathlib import Path


# All tracking files have the same path to  YAML configuration.

DEFAULT_TRACKING_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "tracking.yaml"
)
DEFAULT_BIRTH_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "birth_manager.yaml"
)

__all__ = ["DEFAULT_TRACKING_CONFIG_PATH", "DEFAULT_BIRTH_CONFIG_PATH"]
