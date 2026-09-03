from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


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
