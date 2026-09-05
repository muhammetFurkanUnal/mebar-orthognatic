"""Build a single training manifest from batched images, CVAT XML and TVL CSV files.

To add a new batch, append the corresponding paths to IMAGE_DIRS, XML_FILES and
CSV_FILES at the same index, then run:

    python data/manifest_builder.py

CSV_FILES may be shorter than IMAGE_DIRS/XML_FILES, or contain ``None``/an empty
string for a batch whose TVL labels have not arrived yet.  Ordering remains
index-based in every case.

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

from PIL import Image, ImageOps


# ---------------------------------------------------------------------------
# Batch configuration: update only these three lists when a new batch arrives.
# Entries at the same index must belong to the same batch.
# Paths are relative to the project root (absolute paths are also supported).
# ---------------------------------------------------------------------------
IMAGE_DIRS = [
    "data/raw/images/batch-1-imgs",
    "data/raw/images/batch-2-imgs",
    "data/raw/images/batch-3-imgs",
    "data/raw/images/batch-4-imgs",
]

XML_FILES = [
    "data/raw/annotations/batch-1-annotations.xml",
    "data/raw/annotations/batch-2-annotations.xml",
    "data/raw/annotations/batch-3-annotations.xml",
    "data/raw/annotations/batch-4-annotations.xml",
]

CSV_FILES = [
    "data/raw/TVL-regression/batch-1-TVL-labels.csv",
    "data/raw/TVL-regression/batch-2-TVL-labels.csv",
    "data/raw/TVL-regression/batch-3-TVL-labels.csv",
    None
    # Use None (or "") to reserve an index for labels that have not arrived.
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
    # U+FFFD means the source decoder already lost one character. Preserve it
    # as a one-character wildcard when matching CSV headers; match_csv_name
    # still rejects zero or multiple matches.
    if keep_wildcard:
        value = value.replace("\ufffd", "?")
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


def read_image_dimensions(image_path: Path) -> tuple[int, int, bool]:
    """Read actual image dimensions, applying EXIF orientation correction.

    Returns
    -------
    (width, height, exif_transformed)
        ``exif_transformed`` is ``True`` when the image carried an EXIF
        orientation tag that required a dimension swap (e.g. raw sensor
        data stored as landscape while the photo was shot in portrait).
    """
    with Image.open(image_path) as img:
        raw_w, raw_h = img.size
        transposed = ImageOps.exif_transpose(img)
        if transposed is not None:
            corrected_w, corrected_h = transposed.size
            was_transformed = (raw_w, raw_h) != (corrected_w, corrected_h)
            return corrected_w, corrected_h, was_transformed
        return raw_w, raw_h, False


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


def parse_cvat_xml(
    xml_path: Path, image_keys: Iterable[str]
) -> tuple[dict[str, dict[str, object]], dict[str, list[tuple[str, str]]]]:
    """Read CVAT annotations while retaining per-image XML validation errors."""
    expected = set(image_keys)
    annotations: dict[str, dict[str, object]] = {}
    errors: dict[str, list[tuple[str, str]]] = {image_key: [] for image_key in expected}

    def add_error(image_key: str, error_type: str, message: str) -> None:
        errors[image_key].append((error_type, message))

    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        message = f"Could not parse CVAT XML file {xml_path}: {exc}"
        for image_key in expected:
            add_error(image_key, "xml_parse_error", message)
        return annotations, errors

    for image in root.findall("image"):
        xml_name = image.attrib.get("name", "")
        image_key = normalized_name(xml_name)
        if image_key not in expected:
            continue
        if image_key in annotations:
            add_error(
                image_key,
                "xml_duplicate_image",
                f"Duplicate XML image {xml_name!r} in {xml_path}",
            )
            continue

        record: dict[str, object] = {"cvat_image_name": xml_name}
        try:
            record["width"] = int(image.attrib["width"])
            record["height"] = int(image.attrib["height"])
        except (KeyError, ValueError) as exc:
            add_error(
                image_key,
                "xml_image_metadata_error",
                f"Invalid width/height for image {xml_name!r} in {xml_path}: {exc}",
            )

        seen_labels: set[str] = set()
        valid_labels: set[str] = set()
        for point in image.findall("points"):
            try:
                label = slug(point.attrib["label"])
                point_text = point.attrib["points"]
            except (KeyError, ValueError) as exc:
                add_error(
                    image_key,
                    "xml_point_metadata_error",
                    f"Invalid point metadata for image {xml_name!r} in {xml_path}: {exc}",
                )
                continue
            if label in seen_labels:
                add_error(
                    image_key,
                    "xml_duplicate_label",
                    f"Duplicate point label {label!r} for image {xml_name!r} in {xml_path}",
                )
                continue
            seen_labels.add(label)
            coordinates = point_text.split(";")
            if len(coordinates) != 1:
                add_error(
                    image_key,
                    "xml_multiple_coordinates",
                    f"Expected one coordinate for label {label!r} in image {xml_name!r} "
                    f"within XML file {xml_path}, but found {len(coordinates)} coordinates: "
                    f"{point_text!r}",
                )
                continue
            coordinate_parts = coordinates[0].split(",")
            if len(coordinate_parts) != 2:
                add_error(
                    image_key,
                    "xml_coordinate_format_error",
                    f"Expected an 'x,y' coordinate for label {label!r} in image {xml_name!r} "
                    f"within XML file {xml_path}, but found {coordinates[0]!r}",
                )
                continue
            x_text, y_text = coordinate_parts
            try:
                record[f"{label}_x"] = float(x_text)
                record[f"{label}_y"] = float(y_text)
            except ValueError:
                add_error(
                    image_key,
                    "xml_coordinate_value_error",
                    f"Invalid numeric coordinate for label {label!r} in image {xml_name!r} "
                    f"within XML file {xml_path}: {coordinates[0]!r}",
                )
                continue
            valid_labels.add(label)

        record["landmark_count"] = len(valid_labels)
        annotations[image_key] = record

    for image_key in expected - set(annotations):
        add_error(
            image_key,
            "xml_annotation_missing",
            f"Image has no XML annotation in {xml_path}",
        )
    return annotations, errors


def validate_configuration() -> list[tuple[Path, Path, Path | None]]:
    """Resolve and validate aligned batch configuration lists."""
    if len(IMAGE_DIRS) != len(XML_FILES):
        raise ValueError(
            "IMAGE_DIRS and XML_FILES must have the same length; "
            f"got ({len(IMAGE_DIRS)}, {len(XML_FILES)})"
        )
    if not IMAGE_DIRS:
        raise ValueError("At least one batch must be configured")
    if len(CSV_FILES) > len(IMAGE_DIRS):
        raise ValueError(
            "CSV_FILES cannot contain more entries than IMAGE_DIRS; "
            f"got ({len(CSV_FILES)}, {len(IMAGE_DIRS)})"
        )

    batches = []
    csv_entries = list(CSV_FILES) + [None] * (len(IMAGE_DIRS) - len(CSV_FILES))
    for image_dir_text, xml_text, csv_text in zip(IMAGE_DIRS, XML_FILES, csv_entries):
        image_dir = resolve_path(image_dir_text)
        xml_path = resolve_path(xml_text)
        csv_path = None if csv_text is None or not str(csv_text).strip() else resolve_path(csv_text)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        for path in (xml_path, csv_path):
            if path is None:
                continue
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

        xml_annotations, xml_errors = parse_cvat_xml(xml_path, images)
        tvl_measurements = None
        csv_errors: dict[str, list[tuple[str, str]]] = {image_key: [] for image_key in images}
        if csv_path:
            try:
                tvl_measurements = parse_tvl_csv(csv_path, images)
            except (OSError, UnicodeError, ValueError, csv.Error) as exc:
                message = f"Could not parse TVL CSV file {csv_path}: {exc}"
                for image_key in images:
                    csv_errors[image_key].append(("csv_parse_error", message))

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
                "source_csv": project_relative(csv_path) if csv_path else None,
            }
            record.update(xml_annotations.get(image_key, {}))

            # --- Validate and correct dimensions against the actual image file ---
            try:
                real_w, real_h, exif_xformed = read_image_dimensions(image_path)
                record["exif_transformed"] = exif_xformed
                xml_w = record.get("width")
                xml_h = record.get("height")
                if xml_w is not None and xml_h is not None:
                    if real_w != xml_w or real_h != xml_h:
                        xml_errors[image_key].append(
                            (
                                "dimension_mismatch",
                                f"XML dimensions ({xml_w}x{xml_h}) differ from actual "
                                f"EXIF-corrected image dimensions ({real_w}x{real_h}) "
                                f"for {image_path.name}. Using real dimensions.",
                            )
                        )
                record["width"] = real_w
                record["height"] = real_h
            except Exception as exc:
                xml_errors[image_key].append(
                    (
                        "dimension_read_error",
                        f"Could not read actual image dimensions from "
                        f"{image_path.name}: {exc}",
                    )
                )
                record.setdefault("exif_transformed", False)
            # --------------------------------------------------------------------
            if tvl_measurements:
                record.update(tvl_measurements[image_key])

            record_errors = xml_errors[image_key] + csv_errors[image_key]
            if record_errors:
                record["error_type"] = "|".join(
                    dict.fromkeys(error_type for error_type, _ in record_errors)
                )
                record["error_message"] = " | ".join(
                    message for _, message in record_errors
                )
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
        "exif_transformed",
        "source_xml",
        "source_csv",
        "error_type",
        "error_message",
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
