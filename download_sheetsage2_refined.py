"""Download or check the local SheetSage2-Refined model used by UniversalMMH."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.sheetsage_download import download_refined_model, missing_refined_files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "models/audio_encoders/SheetSage2-Refined",
                        help="Destination model folder; defaults to the standard ComfyUI model directory.")
    parser.add_argument("--check", action="store_true", help="Check local files only; do not download anything.")
    args = parser.parse_args(argv)
    folder = args.model_dir.resolve()
    if args.check:
        missing = missing_refined_files(folder)
        if missing:
            print("SheetSage2-Refined is missing: " + ", ".join(missing))
            return 1
    else:
        download_refined_model(folder)
    print(f"SheetSage2-Refined is ready: {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
