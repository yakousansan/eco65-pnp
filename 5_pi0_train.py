from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "pi0_eco65.yaml"


def main() -> int:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")

    train_cmd = shutil.which("lerobot-train")
    if train_cmd is None:
        raise RuntimeError(
            "Cannot find `lerobot-train` in PATH. Run this with the `vla` conda environment."
        )

    cmd = [train_cmd, "--config_path", str(CONFIG_PATH)]
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
