"""Run the serial finite-volume Burgers solver from the work_4 root."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from methods.fvm_model_serial import main


if __name__ == "__main__":
    main()
