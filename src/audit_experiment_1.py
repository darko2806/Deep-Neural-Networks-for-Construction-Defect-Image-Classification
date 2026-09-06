import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


DATASET_DIRECTORY = Path("MBDD2025")

OUTPUT_PATH = Path(
    "data/results/experiment_1_binary/"
    "dataset_audit.json"
)


def main():
    images_directory = (
        DATASET_DIRECTORY / "JPEGImages"
    )
    annotations_directory = (
        DATASET_DIRECTORY / "Annotations"
    )
    labels_directory = (
        DATASET_DIRECTORY / "Labels"
    )

    image_paths = sorted(
        images_directory.glob("*.jpg")
    )
    xml_paths = sorted(
        annotations_directory.glob("*.xml")
    )
    label_paths = sorted(
        labels_directory.glob("*.txt")
    )

    image_ids = {
        path.stem for path in image_paths
    }
    xml_ids = {
        path.stem for path in xml_paths
    }
    label_ids = {
        path.stem for path in label_paths
    }

    object_counts = Counter()
    images_per_class = Counter()
    unique_class_count_distribution = Counter()

    empty_xml_ids = []
    empty_label_ids = []
    parse_errors = []

    for xml_path in xml_paths:
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as error:
            parse_errors.append(
                {
                    "image_id": xml_path.stem,
                    "error": str(error),
                }
            )
            continue

        classes = []

        for annotation_object in root.findall(
            "object"
        ):
            class_name = annotation_object.findtext(
                "name"
            )

            if class_name is None:
                continue

            class_name = class_name.strip().lower()

            if class_name:
                classes.append(class_name)

        if not classes:
            empty_xml_ids.append(xml_path.stem)

        object_counts.update(classes)

        unique_classes = set(classes)

        unique_class_count_distribution[
            len(unique_classes)
        ] += 1

        for class_name in unique_classes:
            images_per_class[class_name] += 1

    for label_path in label_paths:
        content = label_path.read_text(
            encoding="utf-8"
        ).strip()

        if not content:
            empty_label_ids.append(label_path.stem)

    negative_candidates = sorted(
        set(empty_xml_ids)
        & set(empty_label_ids)
        & image_ids
    )

    audit = {
        "experiment": (
            "experiment_1_binary_"
            "defect_vs_no_defect"
        ),
        "file_counts": {
            "images": len(image_paths),
            "xml_annotations": len(xml_paths),
            "label_files": len(label_paths),
        },
        "consistency": {
            "images_without_xml": sorted(
                image_ids - xml_ids
            ),
            "xml_without_images": sorted(
                xml_ids - image_ids
            ),
            "images_without_label_file": sorted(
                image_ids - label_ids
            ),
            "labels_without_images": sorted(
                label_ids - image_ids
            ),
            "xml_parse_errors": parse_errors,
        },
        "annotation_statistics": {
            "object_counts_by_class": dict(
                sorted(object_counts.items())
            ),
            "images_per_class": dict(
                sorted(images_per_class.items())
            ),
            "images_by_unique_class_count": {
                str(class_count): image_count
                for class_count, image_count
                in sorted(
                    unique_class_count_distribution.items()
                )
            },
        },
        "negative_candidates": {
            "empty_xml_count": len(empty_xml_ids),
            "empty_label_count": len(
                empty_label_ids
            ),
            "candidate_count": len(
                negative_candidates
            ),
            "image_ids": negative_candidates,
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            audit,
            file,
            indent=2,
        )

    print("Eksperiment 1 – audit dataseta")
    print("Broj slika:", len(image_paths))
    print("Broj XML fajlova:", len(xml_paths))
    print(
        "Broj YOLO label fajlova:",
        len(label_paths),
    )
    print(
        "Slike bez XML anotacije:",
        len(image_ids - xml_ids),
    )
    print(
        "Prazne XML anotacije:",
        len(empty_xml_ids),
    )
    print(
        "Prazni label fajlovi:",
        len(empty_label_ids),
    )
    print(
        "Potencijalne No Defect slike:",
        len(negative_candidates),
    )
    print(
        "ID kandidata:",
        negative_candidates,
    )
    print(
        "\nBroj objekata po klasi:",
        dict(sorted(object_counts.items())),
    )
    print(
        "\nBroj slika prema broju klasa:",
        dict(
            sorted(
                unique_class_count_distribution.items()
            )
        ),
    )
    print(
        "\nIzveštaj je sačuvan u:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()