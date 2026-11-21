"""The local adapter is finished when its folder contains the saved matrices."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "outputs" / "lora-adapter"


def main() -> None:
    ready = (ADAPTER / "adapter_config.json").exists()
    print(json.dumps({
        "host": "local",
        "state": "COMPLETED" if ready else "NOT_STARTED",
        "adapter_dir": str(ADAPTER),
    }, indent=2))


if __name__ == "__main__":
    main()
