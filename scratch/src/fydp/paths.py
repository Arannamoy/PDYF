from pathlib import Path
from omegaconf import OmegaConf


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else project_root() / p


def load_data_cfg():
    yaml_path = project_root() / "configs" / "data.yml"
    return OmegaConf.load(yaml_path), yaml_path


if __name__ == "__main__":
    root = project_root()
    cfg, yaml_path = load_data_cfg()

    imu = resolve_path(cfg.data_dir)

    print("paths.py file :", Path(__file__).resolve())
    print("config file   :", yaml_path)
    print("project root  :", root)
    print("root exists   :", root.exists())
    print("IMU path      :", imu)
    print("IMU exists    :", imu.exists())