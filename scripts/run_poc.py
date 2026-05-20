from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermocompute.config import DeviceConfig
from thermocompute.experiments import proof_of_concept_checks


if __name__ == "__main__":
    result = proof_of_concept_checks(DeviceConfig.auto())
    print(json.dumps({"name": result.name, "metrics": result.metrics}, indent=2))
