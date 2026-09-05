"""Convert the project manifest into an Ultralytics YOLO Pose dataset.

The manifest has keypoints but no object bounding boxes. Each image contains one
profile, so the generated pose label uses the full image as its bounding box.
Keypoints are written in anatomical top-to-bottom order and marked visible.

Run from the project root:

    .venv/bin/python ai_models/yolo26s_pose/prepare_dataset.py

Example with explicit options:

    .venv/bin/python ai_models/yolo26s_pose/prepare_dataset.py \\
        --manifest data/processed/manifest.csv \\ (OPTIONAL)
        --output ai_models/yolo26s_pose/yolo_dataset \\ (OPTIONAL)
        --val-fraction 0.20 \\ (OPTIONAL)
        --seed 42 \\ (OPTIONAL)
        --image-mode symlink (OPTIONAL)

Use ``--help`` to see all options. The output directory must be empty.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/processed/manifest.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "yolo_dataset"


KEYPOINTS = (
    "subnasale",
    "upper_lip_anterior",
    "lower_lip_anterior",
    "pogonion",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert data/processed/manifest.csv to YOLO Pose format."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Base directory for relative image_path values in the manifest.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation fraction, split independently within each batch.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How source images are placed in the YOLO dataset.",
    )
    return parser.parse_args()


def read_manifest(path: Path, project_root: Path) -> list[dict[str, object]]:
    required = {"sample_id", "batch", "image_path", "width", "height"}
    required.update(f"{name}_{axis}" for name in KEYPOINTS for axis in ("x", "y"))

    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        samples: list[dict[str, object]] = []
        skipped: list[tuple[str, int]] = []
        seen_ids: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            sample_id = row["sample_id"].strip()
            if not sample_id or sample_id in seen_ids:
                raise ValueError(
                    f"Invalid or duplicate sample_id at manifest line {line_number}: "
                    f"{sample_id!r}"
                )
            seen_ids.add(sample_id)

            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = project_root / image_path
            image_path = image_path.resolve()
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image at manifest line {line_number} does not exist: {image_path}"
                )

            try:
                width = int(row["width"])
                height = int(row["height"])
            except ValueError as error:
                raise ValueError(
                    f"Invalid image size at manifest line {line_number}"
                ) from error

            raw_coordinates = [
                (row[f"{name}_x"].strip(), row[f"{name}_y"].strip())
                for name in KEYPOINTS
            ]
            if any(x == "" or y == "" for x, y in raw_coordinates):
                # Samples without complete annotations are unusable for pose
                # training; skip them instead of failing the whole conversion.
                skipped.append((sample_id, line_number))
                continue

            try:
                coordinates = [(float(x), float(y)) for x, y in raw_coordinates]
            except ValueError as error:
                raise ValueError(
                    f"Invalid numeric value at manifest line {line_number}"
                ) from error

            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image size at manifest line {line_number}")
            for name, (x, y) in zip(KEYPOINTS, coordinates):
                if not (0.0 <= x <= width and 0.0 <= y <= height):
                    raise ValueError(
                        f"Keypoint {name!r} is outside the image at manifest line "
                        f"{line_number}: ({x}, {y}) vs {width}x{height}"
                    )

            samples.append(
                {
                    "sample_id": sample_id,
                    "batch": row["batch"].strip(),
                    "image_path": image_path,
                    "width": width,
                    "height": height,
                    "coordinates": coordinates,
                }
            )

    if not samples:
        raise ValueError(f"Manifest contains no usable samples: {path}")
    if skipped:
        print(
            f"Skipped {len(skipped)} sample(s) with missing keypoint "
            f"coordinates: {[sample_id for sample_id, _ in skipped]}"
        )
    return samples


def assign_splits(
    samples: list[dict[str, object]], val_fraction: float, seed: int
) -> dict[str, str]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("--val-fraction must be in the range [0, 1)")

    by_batch: dict[str, list[dict[str, object]]] = defaultdict(list)
    for sample in samples:
        by_batch[str(sample["batch"])].append(sample)

    splits: dict[str, str] = {}
    for batch_samples in by_batch.values():
        ordered = sorted(
            batch_samples,
            key=lambda sample: hashlib.sha256(
                f"{seed}:{sample['sample_id']}".encode()
            ).digest(),
        )
        val_count = round(len(ordered) * val_fraction)
        if val_fraction > 0 and len(ordered) > 1:
            val_count = max(1, min(val_count, len(ordered) - 1))
        validation_ids = {
            str(sample["sample_id"]) for sample in ordered[:val_count]
        }
        for sample in ordered:
            sample_id = str(sample["sample_id"])
            splits[sample_id] = "val" if sample_id in validation_ids else "train"
    return splits


def pose_label(sample: dict[str, object]) -> str:
    width = int(sample["width"])
    height = int(sample["height"])
    values = ["0", "0.50000000", "0.50000000", "1.00000000", "1.00000000"]
    for x, y in sample["coordinates"]:  # type: ignore[assignment]
        values.extend((f"{x / width:.8f}", f"{y / height:.8f}", "2"))
    return " ".join(values) + "\n"


def place_image(source: Path, destination: Path, mode: str) -> None:
    if mode == "symlink":
        destination.symlink_to(os.path.relpath(source, start=destination.parent))
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def write_yaml(output: Path) -> None:
    keypoint_comment = ", ".join(KEYPOINTS)
    content = (
        f"path: {output.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: profile\n\n"
        f"kpt_shape: [{len(KEYPOINTS)}, 3]\n"
        f"flip_idx: [{', '.join(str(index) for index in range(len(KEYPOINTS)))}]\n"
        f"# keypoint order: {keypoint_comment}\n"
    )
    (output / "data.yaml").write_text(content, encoding="utf-8")


def convert(args: argparse.Namespace) -> dict[str, int]:
    manifest = args.manifest.resolve()
    output = args.output.resolve()
    samples = read_manifest(manifest, args.project_root.resolve())
    splits = assign_splits(samples, args.val_fraction, args.seed)

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}. Choose a new --output path."
        )
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    mapping_rows: list[dict[str, str]] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        split = splits[sample_id]
        source = Path(sample["image_path"])
        image_name = f"{sample_id}{source.suffix.lower()}"
        place_image(source, output / "images" / split / image_name, args.image_mode)
        (output / "labels" / split / f"{sample_id}.txt").write_text(
            pose_label(sample), encoding="utf-8"
        )
        mapping_rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "source_image": source.as_posix(),
                "yolo_image": f"images/{split}/{image_name}",
            }
        )
        counts[split] += 1

    with (output / "samples.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=("sample_id", "split", "source_image", "yolo_image")
        )
        writer.writeheader()
        writer.writerows(mapping_rows)
    write_yaml(output)
    return counts


def main() -> None:
    args = parse_args()
    counts = convert(args)
    print(f"YOLO dataset written to: {args.output.resolve()}")
    print(f"Train samples: {counts['train']}")
    print(f"Validation samples: {counts['val']}")
    print(f"Dataset config: {(args.output.resolve() / 'data.yaml')}")


if __name__ == "__main__":
    main()
