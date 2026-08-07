"""Build a single training manifest from batched images, CVAT XML and TVL CSV files.

To add a new batch, append the corresponding paths to IMAGE_DIRS, XML_FILES and
CSV_FILES at the same index, then run:

    python data/manifest_builder.py

Only metadata is combined; images remain untouched in their raw directories.
"""

from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Batch configuration: update only these three lists when a new batch arrives.
# Entries at the same index must belong to the same batch.
# Paths are relative to the project root (absolute paths are also supported).
# ---------------------------------------------------------------------------
IMAGE_DIRS = [
    "data/raw/batch-1-imgs",
    "data/raw/batch-2-imgs",
]

XML_FILES = [
    "data/raw/annotations batch 1 .xml",
    "data/raw/annotations batch 2.xml",
]

CSV_FILES = [
    "data/raw/batch-1-TVL-labels.csv",
    "data/raw/batch-2-TVL-labels.csv",
]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "data/processed/manifest.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_path(path: str | Path) -> Path:
    """Resolve configuration paths relative to the project root."""
    path = Path(path)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def project_relative(path: Path) -> str:
    """Prefer portable project-relative paths in the generated manifest."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def strip_image_extensions(name: str) -> str:
    """Strip repeated image extensions such as '.jpg.jpg' or '.jpg.png'."""
    result = Path(name).name
    while Path(result).suffix.lower() in IMAGE_EXTENSIONS:
        result = result[: -len(Path(result).suffix)]
    return result


def normalized_name(name: str, *, keep_wildcard: bool = False) -> str:
    """Normalize names for matching without changing the original files."""
    value = unicodedata.normalize("NFKD", strip_image_extensions(name).casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    # Treat Turkish dotted/dotless I identically. Some batch-2 CSV characters
    # were already replaced with '?' by the source export.
    value = value.replace("ı", "i")
    allowed = {"?"} if keep_wildcard else set()
    return "".join(char for char in value if char.isalnum() or char in allowed)


def slug(value: str) -> str:
    """Create a stable ASCII-like column name."""
    result = unicodedata.normalize("NFKD", value.casefold())
    result = "".join(char for char in result if not unicodedata.combining(char))
    result = result.replace("ı", "i")
    result = re.sub(r"[^a-z0-9]+", "_", result).strip("_")
    if not result:
        raise ValueError(f"Cannot create a column name from {value!r}")
    return result


def stable_sample_id(batch: str, image_name: str) -> str:
    """Generate a deterministic ID without exposing a person's name."""
    identity = f"{batch}/{normalized_name(image_name)}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"sample_{digest}"


def find_images(image_dir: Path) -> dict[str, Path]:
    """Index supported image files and reject ambiguous normalized names."""
    images: dict[str, Path] = {}
    for path in sorted(image_dir.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = normalized_name(path.name)
        if key in images:
            raise ValueError(
                f"Ambiguous image names in {image_dir}: "
                f"{images[key].name!r} and {path.name!r}"
            )
        images[key] = path
    return images


def read_text_with_fallback(path: Path) -> str:
    """Decode CSV exports commonly produced with UTF-8 or Turkish code pages."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1254", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Could not decode {path}")


def parse_number(value: str, *, source: Path) -> float | None:
    """Parse ordinary numbers and the source notation '(-)0.115'."""
    value = value.strip()
    if not value:
        return None
    if value.startswith("(-)"):
        value = "-" + value[3:]
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"Invalid numeric value {value!r} in {source}") from error


def wildcard_matches(pattern: str, candidates: Iterable[str]) -> list[str]:
    """Match CSV names where '?' represents one lost source character."""
    expression = "^" + re.escape(pattern).replace(r"\?", ".") + "$"
    return [candidate for candidate in candidates if re.fullmatch(expression, candidate)]


def match_csv_name(csv_name: str, image_keys: Iterable[str], csv_path: Path) -> str:
    """Return exactly one image key for a CSV header, or fail loudly."""
    image_keys = list(image_keys)
    pattern = normalized_name(csv_name, keep_wildcard=True)
    matches = wildcard_matches(pattern, image_keys)
    if len(matches) != 1:
        raise ValueError(
            f"CSV header {csv_name!r} in {csv_path} matched {len(matches)} images. "
            f"Matches: {matches}"
        )
    return matches[0]


def parse_tvl_csv(csv_path: Path, image_keys: Iterable[str]) -> dict[str, dict[str, float]]:
    """Transpose the source CSV so each image receives its TVL measurements."""
    text = read_text_with_fallback(csv_path)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(text.splitlines(), dialect))
    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    headers = rows[0]
    measurements: dict[str, dict[str, float]] = {}
    for column_index, csv_name in enumerate(headers[1:], start=1):
        if not csv_name.strip():
            continue
        image_key = match_csv_name(csv_name, image_keys, csv_path)
        values: dict[str, float] = {}
        for row in rows[1:]:
            if not row or not row[0].strip():
                continue
            if column_index >= len(row):
                raise ValueError(f"Missing value for {csv_name!r} in {csv_path}")
            number = parse_number(row[column_index], source=csv_path)
            if number is not None:
                values[slug(row[0])] = number
        measurements[image_key] = values

    expected = set(image_keys)
    if set(measurements) != expected:
        missing = sorted(expected - set(measurements))
        raise ValueError(f"Images without CSV measurements in {csv_path}: {missing}")
    return measurements


def parse_cvat_xml(xml_path: Path, image_keys: Iterable[str]) -> dict[str, dict[str, object]]:
    """Read image dimensions and point annotations from a CVAT XML export."""
    expected = set(image_keys)
    annotations: dict[str, dict[str, object]] = {}
    root = ET.parse(xml_path).getroot()

    for image in root.findall("image"):
        xml_name = image.attrib.get("name", "")
        image_key = normalized_name(xml_name)
        if image_key not in expected:
            raise ValueError(f"XML image {xml_name!r} in {xml_path} has no matching file")
        if image_key in annotations:
            raise ValueError(f"Duplicate XML image {xml_name!r} in {xml_path}")

        record: dict[str, object] = {
            "cvat_image_name": xml_name,
            "width": int(image.attrib["width"]),
            "height": int(image.attrib["height"]),
        }
        seen_labels: set[str] = set()
        for point in image.findall("points"):
            label = slug(point.attrib["label"])
            if label in seen_labels:
                raise ValueError(f"Duplicate point label {label!r} for {xml_name!r}")
            seen_labels.add(label)
            coordinates = point.attrib["points"].split(";")
            if len(coordinates) != 1:
                raise ValueError(
                    f"Expected one coordinate for {label!r} in {xml_name!r}, "
                    f"found {len(coordinates)}"
                )
            x_text, y_text = coordinates[0].split(",")
            record[f"{label}_x"] = float(x_text)
            record[f"{label}_y"] = float(y_text)

        record["landmark_count"] = len(seen_labels)
        annotations[image_key] = record

    if set(annotations) != expected:
        missing = sorted(expected - set(annotations))
        raise ValueError(f"Images without XML annotations in {xml_path}: {missing}")
    return annotations


def validate_configuration() -> list[tuple[Path, Path, Path]]:
    """Resolve and validate aligned batch configuration lists."""
    lengths = (len(IMAGE_DIRS), len(XML_FILES), len(CSV_FILES))
    if len(set(lengths)) != 1:
        raise ValueError(
            "IMAGE_DIRS, XML_FILES and CSV_FILES must have the same length; "
            f"got {lengths}"
        )
    if not IMAGE_DIRS:
        raise ValueError("At least one batch must be configured")

    batches = []
    for image_dir_text, xml_text, csv_text in zip(IMAGE_DIRS, XML_FILES, CSV_FILES):
        image_dir = resolve_path(image_dir_text)
        xml_path = resolve_path(xml_text)
        csv_path = resolve_path(csv_text)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        for path in (xml_path, csv_path):
            if not path.is_file():
                raise FileNotFoundError(f"Input file does not exist: {path}")
        batches.append((image_dir, xml_path, csv_path))
    return batches


def build_manifest() -> list[dict[str, object]]:
    """Build and validate manifest records for all configured batches."""
    records: list[dict[str, object]] = []
    sample_ids: set[str] = set()

    for image_dir, xml_path, csv_path in validate_configuration():
        batch = image_dir.name
        images = find_images(image_dir)
        if not images:
            raise ValueError(f"No supported images found in {image_dir}")

        xml_annotations = parse_cvat_xml(xml_path, images)
        tvl_measurements = parse_tvl_csv(csv_path, images)

        for image_key, image_path in sorted(images.items()):
            sample_id = stable_sample_id(batch, image_path.name)
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate sample_id generated: {sample_id}")
            sample_ids.add(sample_id)

            record: dict[str, object] = {
                "sample_id": sample_id,
                "batch": batch,
                "image_path": project_relative(image_path),
                "image_name": image_path.name,
                "source_xml": project_relative(xml_path),
                "source_csv": project_relative(csv_path),
            }
            record.update(xml_annotations[image_key])
            record.update(tvl_measurements[image_key])
            records.append(record)

    return records


def write_manifest(records: list[dict[str, object]]) -> None:
    """Write records atomically enough for normal local use."""
    fixed_columns = [
        "sample_id",
        "batch",
        "image_path",
        "image_name",
        "width",
        "height",
        "landmark_count",
        "cvat_image_name",
        "source_xml",
        "source_csv",
    ]
    extra_columns = sorted(set().union(*(record.keys() for record in records)) - set(fixed_columns))
    fieldnames = fixed_columns + extra_columns

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = OUTPUT_FILE.with_suffix(OUTPUT_FILE.suffix + ".tmp")
    with temporary_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    temporary_file.replace(OUTPUT_FILE)


def main() -> None:
    records = build_manifest()
    write_manifest(records)
    batch_counts: dict[str, int] = {}
    for record in records:
        batch = str(record["batch"])
        batch_counts[batch] = batch_counts.get(batch, 0) + 1

    print(f"Manifest written to: {project_relative(OUTPUT_FILE)}")
    print(f"Total samples: {len(records)}")
    for batch, count in sorted(batch_counts.items()):
        print(f"  {batch}: {count}")


if __name__ == "__main__":
    main()
