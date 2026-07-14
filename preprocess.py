#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from pathlib import Path
import argparse

import pandas as pd
from PIL import Image
from tqdm import tqdm

from cli_completion import parse_args_with_completion


def preprocess_image(
    img: Image.Image,
    crop_y_start: int,
    out_width: int,
    out_height: int,
) -> Image.Image:
    img = img.convert("RGB")

    width, height = img.size
    crop_y_start = max(0, min(crop_y_start, height - 1))

    # Unteren Bildbereich behalten
    img = img.crop((0, crop_y_start, width, height))

    # Auf Trainingsauflösung bringen
    img = img.resize((out_width, out_height), Image.BILINEAR)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-run",
        type=str,
        required=True,
        help=(
            "Pfad zum Run-Ordner, z.B. "
            "~/duckietown/data/imitation_learning/train/run_001_20260707_094333"
        ),
    )
    parser.add_argument("--crop-y-start", type=int, default=200)
    parser.add_argument("--out-width", type=int, default=224)
    parser.add_argument("--out-height", type=int, default=224)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Existierende Dateien in images_processed überschreiben",
    )
    args = parse_args_with_completion(parser)

    run_dir = Path(args.input_run)
    input_images_dir = run_dir / "images"
    output_images_dir = run_dir / "images_processed"
    actions_csv = run_dir / "actions.csv"

    if not run_dir.exists():
        raise FileNotFoundError(f"Run-Ordner nicht gefunden: {run_dir}")

    if not input_images_dir.exists():
        raise FileNotFoundError(f"Input-Bildordner nicht gefunden: {input_images_dir}")

    if not actions_csv.exists():
        raise FileNotFoundError(f"actions.csv nicht gefunden: {actions_csv}")

    output_images_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(actions_csv)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing images"):
        image_rel = Path(row["image"])      # z.B. images/000123.jpg
        image_name = image_rel.name         # 000123.jpg

        input_img_path = input_images_dir / image_name
        output_img_path = output_images_dir / image_name

        if output_img_path.exists() and not args.overwrite:
            continue

        if not input_img_path.exists():
            raise FileNotFoundError(f"Bild nicht gefunden: {input_img_path}")

        with Image.open(input_img_path) as img:
            img_processed = preprocess_image(
                img,
                crop_y_start=args.crop_y_start,
                out_width=args.out_width,
                out_height=args.out_height,
            )

            img_processed.save(
                output_img_path,
                format="JPEG",
                quality=args.jpeg_quality,
            )

    print(f"Fertig.")
    print(f"Input-Bilder:  {input_images_dir}")
    print(f"Output-Bilder: {output_images_dir}")
    print(f"actions.csv bleibt unverändert: {actions_csv}")


if __name__ == "__main__":
    main()
