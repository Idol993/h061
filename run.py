from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from active_learner.cli import main

if __name__ == "__main__":
    main()
