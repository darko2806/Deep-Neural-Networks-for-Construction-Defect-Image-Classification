import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd

from sklearn.model_selection import train_test_split

def parse_annotation(xml_path):
    xml_path = Path(xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")

    width = int(float(size.findtext("width")))
    height = int(float(size.findtext("height")))

    objects = root.findall("object")

    classes = []

    for obj in objects:
        class_name = obj.findtext("name")

        if class_name is not None:
            class_name = class_name.strip().lower()
            classes.append(class_name)

    unique_classes = sorted(set(classes))

    annotation = {
        "image_id": xml_path.stem,
        "width": width,
        "height": height,
        "n_objects": len(classes),
        "classes": classes,
        "unique_classes": unique_classes,
        "n_classes": len(unique_classes),
    }

    return annotation


def build_metadata(dataset_dir):

    dataset_dir = Path(dataset_dir)

    annotations_dir = dataset_dir / "Annotations"
    images_dir = dataset_dir / "JPEGImages"

    xml_files = sorted(annotations_dir.glob("*.xml"))

    records = []

    for xml_path in xml_files:

        annotation = parse_annotation(xml_path)

        image_path = images_dir / f"{annotation['image_id']}.jpg"

        if annotation["n_classes"] == 1:
            target_class = annotation["unique_classes"][0]
        else:
            target_class = None

        record = {
            "image_id": annotation["image_id"],
            "image_path": str(image_path),
            "width": annotation["width"],
            "height": annotation["height"],
            "n_objects": annotation["n_objects"],
            "n_classes": annotation["n_classes"],
            "classes": "|".join(annotation["unique_classes"]),
            "target_class": target_class,
        }

        records.append(record)

    metadata = pd.DataFrame(records)

    return metadata

def select_single_class_images(metadata):
    single_class_metadata = metadata.loc[
        metadata["target_class"].notna()
    ].copy()

    single_class_metadata = single_class_metadata.reset_index(drop=True)

    return single_class_metadata

def compute_file_hash(file_path, chunk_size=1024 * 1024):
    file_path = Path(file_path)
    file_hash = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            file_hash.update(chunk)

    return file_hash.hexdigest()


def add_content_hashes(metadata):
    metadata_with_hashes = metadata.copy()

    metadata_with_hashes["content_hash"] = (
        metadata_with_hashes["image_path"]
        .apply(compute_file_hash)
    )

    return metadata_with_hashes

def assign_data_splits(
    metadata,
    train_size=0.8,
    validation_size=0.1,
    test_size=0.1,
    random_state=42,
):
    split_sizes = train_size + validation_size + test_size

    if abs(split_sizes - 1.0) > 1e-8:
        raise ValueError(
            "train_size, validation_size i test_size moraju imati zbir 1.0"
        )

    required_columns = {"target_class", "content_hash"}
    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            f"Nedostaju obavezne kolone: {sorted(missing_columns)}"
        )

    if metadata[list(required_columns)].isna().any().any():
        raise ValueError(
            "target_class i content_hash ne smeju sadržati nedostajuće vrednosti"
        )

    classes_per_hash = (
        metadata.groupby("content_hash")["target_class"]
        .nunique()
    )

    if (classes_per_hash > 1).any():
        raise ValueError(
            "Identične slike ne smeju imati različite target_class vrednosti"
        )

    content_groups = (
        metadata[
            ["content_hash", "target_class"]
        ]
        .drop_duplicates(subset=["content_hash"])
        .reset_index(drop=True)
    )

    train_groups, remaining_groups = train_test_split(
        content_groups,
        test_size=validation_size + test_size,
        stratify=content_groups["target_class"],
        random_state=random_state,
    )

    relative_test_size = test_size / (validation_size + test_size)

    validation_groups, test_groups = train_test_split(
        remaining_groups,
        test_size=relative_test_size,
        stratify=remaining_groups["target_class"],
        random_state=random_state,
    )

    train_groups = train_groups.assign(split="train")
    validation_groups = validation_groups.assign(split="validation")
    test_groups = test_groups.assign(split="test")

    split_mapping = (
        pd.concat(
            [train_groups, validation_groups, test_groups],
            ignore_index=True,
        )
        .set_index("content_hash")["split"]
    )

    metadata_with_splits = metadata.copy()
    metadata_with_splits["split"] = (
        metadata_with_splits["content_hash"]
        .map(split_mapping)
    )

    return metadata_with_splits