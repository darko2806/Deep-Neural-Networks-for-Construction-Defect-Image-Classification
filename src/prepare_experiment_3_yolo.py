import json
from pathlib import Path

import pandas as pd


METADATA_PATH = Path("data/metadata.csv")
SOURCE_LABEL_DIRECTORY = Path("MBDD2025/Labels")

YOLO_ROOT = Path("data/yolo_experiment_3")
RESULT_DIRECTORY = Path(
    "data/results/experiment_3_cascade"
)

CLASS_TO_ORIGINAL_ID = {
    "crack": 0,
    "leakage": 1,
    "abscission": 2,
    "corrosion": 3,
    "bulge": 4,
}

SPLIT_DIRECTORY_NAMES = {
    "train": "train",
    "validation": "validation",
    "test": "test",
}


def validate_metadata(metadata):
    required_columns = {
        "image_id",
        "image_path",
        "target_class",
        "visual_group",
        "split",
    }

    missing_columns = (
        required_columns - set(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Nedostaju kolone: {sorted(missing_columns)}"
        )

    expected_splits = {
        "train",
        "validation",
        "test",
    }

    found_splits = set(
        metadata["split"].unique()
    )

    if found_splits != expected_splits:
        raise ValueError(
            f"Pronađeni splitovi: {sorted(found_splits)}"
        )

    expected_classes = set(
        CLASS_TO_ORIGINAL_ID
    )
    found_classes = set(
        metadata["target_class"].unique()
    )

    if found_classes != expected_classes:
        raise ValueError(
            f"Pronađene klase: {sorted(found_classes)}"
        )

    if metadata["image_id"].duplicated().any():
        raise ValueError(
            "Metadata sadrži duplirane image_id vrednosti"
        )

    group_split_counts = (
        metadata.groupby("visual_group")[
            "split"
        ].nunique()
    )

    if group_split_counts.max() != 1:
        raise ValueError(
            "Otkriven je visual_group u više splitova"
        )


def parse_and_remap_label(
    label_path,
    expected_original_class_id,
):
    if not label_path.is_file():
        raise FileNotFoundError(
            f"Nedostaje labela: {label_path}"
        )

    output_lines = []

    for line_number, line in enumerate(
        label_path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"Neispravna YOLO labela: "
                f"{label_path}, red {line_number}"
            )

        original_class_id = int(parts[0])

        if (
            original_class_id
            != expected_original_class_id
        ):
            raise ValueError(
                f"Neočekivana klasa u {label_path}: "
                f"{original_class_id}, očekivano "
                f"{expected_original_class_id}"
            )

        coordinates = [
            float(value)
            for value in parts[1:]
        ]

        x_center, y_center, width, height = (
            coordinates
        )

        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            raise ValueError(
                f"Koordinate van opsega u "
                f"{label_path}, red {line_number}"
            )

        remapped_coordinates = " ".join(
            f"{value:.10f}"
            for value in coordinates
        )

        # Svaki class-specific detektor ima
        # jednu lokalnu klasu sa indeksom 0.
        output_lines.append(
            f"0 {remapped_coordinates}"
        )

    if not output_lines:
        raise ValueError(
            f"Slika nema bounding box: {label_path}"
        )

    return output_lines


def create_image_link(
    source_path,
    destination_path,
):
    source_path = source_path.resolve()

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Nedostaje slika: {source_path}"
        )

    if destination_path.is_symlink():
        if (
            destination_path.resolve()
            != source_path
        ):
            raise ValueError(
                "Postojeći link pokazuje na drugu sliku: "
                f"{destination_path}"
            )

        return

    if destination_path.exists():
        raise FileExistsError(
            "Odredišna putanja već postoji i nije link: "
            f"{destination_path}"
        )

    destination_path.symlink_to(
        source_path
    )


def write_dataset_yaml(
    class_name,
    class_directory,
):
    yaml_path = (
        class_directory / "dataset.yaml"
    )

    text = "\n".join(
        [
            f"path: {class_directory.resolve().as_posix()}",
            "train: images/train",
            "val: images/validation",
            "test: images/test",
            "",
            "names:",
            f"  0: {class_name}",
            "",
        ]
    )

    yaml_path.write_text(
        text,
        encoding="utf-8",
    )

    return yaml_path


def main():
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Nedostaje: {METADATA_PATH}"
        )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    validate_metadata(metadata)

    YOLO_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    RESULT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_records = []
    class_reports = {}

    print(
        "Eksperiment 3 – priprema class-specific "
        "YOLO skupova"
    )
    print("Ukupno slika:", len(metadata))

    for (
        class_name,
        original_class_id,
    ) in CLASS_TO_ORIGINAL_ID.items():
        class_directory = (
            YOLO_ROOT / class_name
        )

        for split_directory in (
            SPLIT_DIRECTORY_NAMES.values()
        ):
            (
                class_directory
                / "images"
                / split_directory
            ).mkdir(
                parents=True,
                exist_ok=True,
            )
            (
                class_directory
                / "labels"
                / split_directory
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

        class_metadata = metadata.loc[
            metadata["target_class"].eq(
                class_name
            )
        ].copy()

        split_image_counts = {}
        split_box_counts = {}

        for split_name, split_directory in (
            SPLIT_DIRECTORY_NAMES.items()
        ):
            split_metadata = class_metadata.loc[
                class_metadata["split"].eq(
                    split_name
                )
            ]

            split_image_counts[split_name] = int(
                len(split_metadata)
            )
            split_box_counts[split_name] = 0

            for row in split_metadata.itertuples(
                index=False
            ):
                source_image_path = Path(
                    row.image_path
                )
                source_label_path = (
                    SOURCE_LABEL_DIRECTORY
                    / f"{row.image_id}.txt"
                )

                image_destination = (
                    class_directory
                    / "images"
                    / split_directory
                    / source_image_path.name
                )
                label_destination = (
                    class_directory
                    / "labels"
                    / split_directory
                    / f"{row.image_id}.txt"
                )

                remapped_lines = (
                    parse_and_remap_label(
                        source_label_path,
                        original_class_id,
                    )
                )

                create_image_link(
                    source_image_path,
                    image_destination,
                )

                label_destination.write_text(
                    "\n".join(
                        remapped_lines
                    ) + "\n",
                    encoding="utf-8",
                )

                number_of_boxes = len(
                    remapped_lines
                )
                split_box_counts[
                    split_name
                ] += number_of_boxes

                manifest_records.append(
                    {
                        "detector_class": (
                            class_name
                        ),
                        "image_id": row.image_id,
                        "split": split_name,
                        "visual_group": (
                            row.visual_group
                        ),
                        "source_image_path": str(
                            source_image_path
                        ),
                        "yolo_image_path": str(
                            image_destination
                        ),
                        "yolo_label_path": str(
                            label_destination
                        ),
                        "number_of_boxes": (
                            number_of_boxes
                        ),
                    }
                )

        yaml_path = write_dataset_yaml(
            class_name,
            class_directory,
        )

        class_reports[class_name] = {
            "original_class_id": int(
                original_class_id
            ),
            "local_detector_class_id": 0,
            "image_counts": (
                split_image_counts
            ),
            "box_counts": split_box_counts,
            "dataset_yaml": str(yaml_path),
        }

        print(f"\n{class_name}")
        print(
            "Slike:",
            split_image_counts,
        )
        print(
            "Bounding box-evi:",
            split_box_counts,
        )
        print("YAML:", yaml_path)

    manifest = pd.DataFrame(
        manifest_records
    )

    if manifest.duplicated(
        subset=[
            "detector_class",
            "image_id",
        ]
    ).any():
        raise ValueError(
            "Manifest sadrži duplirane slike"
        )

    if (
        manifest.groupby(
            [
                "detector_class",
                "visual_group",
            ]
        )["split"]
        .nunique()
        .max()
        != 1
    ):
        raise ValueError(
            "Visual group se pojavljuje u više splitova"
        )

    manifest_path = (
        RESULT_DIRECTORY
        / "detection_dataset_manifest.csv"
    )
    report_path = (
        RESULT_DIRECTORY
        / "detection_dataset_report.json"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    report = {
        "experiment": (
            "experiment_3_cascade_detection"
        ),
        "design": (
            "experiment_2_classifier_routes_image_"
            "to_class_specific_yolo_detector"
        ),
        "source_metadata": str(
            METADATA_PATH
        ),
        "single_class_images_only": True,
        "total_images": int(len(metadata)),
        "class_mapping": (
            CLASS_TO_ORIGINAL_ID
        ),
        "detectors": class_reports,
        "leakage_checks": {
            "same_splits_as_experiment_2": True,
            "one_split_per_visual_group": True,
            "test_not_used_for_training": True,
        },
        "manifest_path": str(
            manifest_path
        ),
    }

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nPriprema je završena.")
    print("Manifest:", manifest_path)
    print("Izveštaj:", report_path)
    print(
        "Provera visual_group curenja: uspešna"
    )


if __name__ == "__main__":
    main()