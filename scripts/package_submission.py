import sys
import zipfile
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import RENDERED_DIR, SUBMISSION_ZIP


def package_submission(rendered_dir, output_zip):
    rendered_dir = Path(rendered_dir)
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for scene_dir in sorted(rendered_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            for image_path in sorted(scene_dir.iterdir()):
                arcname = f"{scene_dir.name}/{image_path.name}"
                zf.write(image_path, arcname)

    return output_zip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rendered_dir", default=str(RENDERED_DIR))
    parser.add_argument("--output", default=str(SUBMISSION_ZIP))
    args = parser.parse_args()

    output_zip = package_submission(args.rendered_dir, args.output)
    print(f"Da dong goi submission tai {output_zip}")


if __name__ == "__main__":
    main()
