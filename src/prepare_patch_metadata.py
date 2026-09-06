import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from matplotlib.pylab import rand
import numpy as np
import pandas as pd


from src.visual_groups import (
    add_perceptual_hashes,
    assign_visual_groups,
)


DATASET_DIRECTORY = Path("MBDD2025")

OUTPUT_PATH = Path(
    "data/patch_metadata.csv"
)

REPORT_PATH = Path(
    "data/results/experiment_1_binary/"
    "patch_metadata_report.json"
)

MAX_PAIRS_PER_IMAGE = 2
POSITIVE_CONTEXT_RATIO = 0.25
MIN_PATCH_SIDE = 96
MAX_PATCH_SIDE = 384
NEGATIVE_EXCLUSION_RATIO = 0.20
MAX_NEGATIVE_ATTEMPTS = 300
RANDOM_STATE = 42

N_SPLITS = 10
VALIDATION_FOLD = 0
TEST_FOLD = 1


def read_annotation(xml_path):
    root = ET.parse(xml_path).getroot()

    size = root.find("size")

    width = int(float(size.findtext("width")))
    height = int(float(size.findtext("height")))

    boxes = []

    for annotation_object in root.findall(
        "object"
    ):
        class_name = annotation_object.findtext(
            "name"
        )

        bounding_box = annotation_object.find(
            "bndbox"
        )

        if class_name is None or bounding_box is None:
            continue

        class_name = class_name.strip().lower()

        xmin = int(
            float(bounding_box.findtext("xmin"))
        ) - 1
        ymin = int(
            float(bounding_box.findtext("ymin"))
        ) - 1
        xmax = int(
            float(bounding_box.findtext("xmax"))
        )
        ymax = int(
            float(bounding_box.findtext("ymax"))
        )

        xmin = max(0, min(xmin, width - 1))
        ymin = max(0, min(ymin, height - 1))
        xmax = max(1, min(xmax, width))
        ymax = max(1, min(ymax, height))

        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append(
            {
                "class_name": class_name,
                "box": (
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                ),
            }
        )

    return {
        "width": width,
        "height": height,
        "boxes": boxes,
    }


def expand_box(
    box,
    image_width,
    image_height,
    ratio,
):
    xmin, ymin, xmax, ymax = box

    box_width = xmax - xmin
    box_height = ymax - ymin

    horizontal_padding = box_width * ratio
    vertical_padding = box_height * ratio

    expanded_xmin = max(
        0,
        math.floor(xmin - horizontal_padding),
    )
    expanded_ymin = max(
        0,
        math.floor(ymin - vertical_padding),
    )
    expanded_xmax = min(
        image_width,
        math.ceil(xmax + horizontal_padding),
    )
    expanded_ymax = min(
        image_height,
        math.ceil(ymax + vertical_padding),
    )

    return (
        expanded_xmin,
        expanded_ymin,
        expanded_xmax,
        expanded_ymax,
    )

def make_positive_patch_box(
    box,
    image_width,
    image_height,
):
    xmin, ymin, xmax, ymax = box

    box_width = xmax - xmin
    box_height = ymax - ymin

    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    required_side = math.ceil(
        max(box_width, box_height)
        * (1 + 2 * POSITIVE_CONTEXT_RATIO)
    )

    patch_side = min(
        max(required_side, MIN_PATCH_SIDE),
        MAX_PATCH_SIDE,
        image_width,
        image_height,
    )

    patch_side = int(patch_side)

    patch_xmin = round(
        center_x - patch_side / 2
    )
    patch_ymin = round(
        center_y - patch_side / 2
    )

    patch_xmin = max(
        0,
        min(
            patch_xmin,
            image_width - patch_side,
        ),
    )

    patch_ymin = max(
        0,
        min(
            patch_ymin,
            image_height - patch_side,
        ),
    )

    patch_xmax = patch_xmin + patch_side
    patch_ymax = patch_ymin + patch_side

    return (
        int(patch_xmin),
        int(patch_ymin),
        int(patch_xmax),
        int(patch_ymax),
    )

def boxes_intersect(first_box, second_box):
    first_xmin, first_ymin, first_xmax, first_ymax = (
        first_box
    )
    (
        second_xmin,
        second_ymin,
        second_xmax,
        second_ymax,
    ) = second_box

    return (
        first_xmin < second_xmax
        and first_xmax > second_xmin
        and first_ymin < second_ymax
        and first_ymax > second_ymin
    )


def sample_negative_box(
    crop_width,
    crop_height,
    image_width,
    image_height,
    forbidden_boxes,
    reference_box,
    random_generator,
):
    if (
        crop_width > image_width
        or crop_height > image_height
    ):
        return None

    maximum_xmin = image_width - crop_width
    maximum_ymin = image_height - crop_height

    (
        reference_xmin,
        reference_ymin,
        reference_xmax,
        reference_ymax,
    ) = reference_box

    reference_center_x = (
        reference_xmin + reference_xmax
    ) / 2

    reference_center_y = (
        reference_ymin + reference_ymax
    ) / 2

    candidate_positions = [
        (
            reference_xmin - crop_width,
            reference_ymin,
        ),
        (
            reference_xmax,
            reference_ymin,
        ),
        (
            reference_xmin,
            reference_ymin - crop_height,
        ),
        (
            reference_xmin,
            reference_ymax,
        ),
        (
            reference_xmin - crop_width,
            reference_ymin - crop_height,
        ),
        (
            reference_xmax,
            reference_ymin - crop_height,
        ),
        (
            reference_xmin - crop_width,
            reference_ymax,
        ),
        (
            reference_xmax,
            reference_ymax,
        ),
    ]

    for _ in range(MAX_NEGATIVE_ATTEMPTS):
        candidate_positions.append(
            (
                int(
                    random_generator.integers(
                        0,
                        maximum_xmin + 1,
                    )
                ),
                int(
                    random_generator.integers(
                        0,
                        maximum_ymin + 1,
                    )
                ),
            )
        )

    valid_candidates = []
    seen_candidates = set()

    for candidate_xmin, candidate_ymin in (
        candidate_positions
    ):
        candidate_xmin = int(
            max(
                0,
                min(
                    candidate_xmin,
                    maximum_xmin,
                ),
            )
        )

        candidate_ymin = int(
            max(
                0,
                min(
                    candidate_ymin,
                    maximum_ymin,
                ),
            )
        )

        candidate_box = (
            candidate_xmin,
            candidate_ymin,
            candidate_xmin + crop_width,
            candidate_ymin + crop_height,
        )

        if candidate_box in seen_candidates:
            continue

        seen_candidates.add(candidate_box)

        intersects_defect = any(
            boxes_intersect(
                candidate_box,
                forbidden_box,
            )
            for forbidden_box in forbidden_boxes
        )

        if intersects_defect:
            continue

        candidate_center_x = (
            candidate_box[0] + candidate_box[2]
        ) / 2

        candidate_center_y = (
            candidate_box[1] + candidate_box[3]
        ) / 2

        center_distance = math.hypot(
            candidate_center_x
            - reference_center_x,
            candidate_center_y
            - reference_center_y,
        )

        valid_candidates.append(
            (
                center_distance,
                candidate_box,
            )
        )

    if not valid_candidates:
        return None

    valid_candidates.sort(
        key=lambda candidate: candidate[0]
    )

    return valid_candidates[0][1]


def assign_patch_splits(patch_metadata):
    group_sizes = (
        patch_metadata
        .groupby(
            "visual_group",
            sort=False,
        )
        .size()
        .rename("patch_count")
        .reset_index()
    )

    random_generator = np.random.default_rng(
        RANDOM_STATE
    )

    # Nasumično razrešavanje grupa iste veličine.
    group_sizes["tie_breaker"] = (
        random_generator.random(
            len(group_sizes)
        )
    )

    # Najveće grupe raspoređujemo prve.
    group_sizes = group_sizes.sort_values(
        [
            "patch_count",
            "tie_breaker",
        ],
        ascending=[
            False,
            True,
        ],
    )

    fold_sizes = np.zeros(
        N_SPLITS,
        dtype=np.int64,
    )

    group_to_fold = {}

    for row in group_sizes.itertuples(
        index=False
    ):
        smallest_fold_size = fold_sizes.min()

        candidate_folds = np.flatnonzero(
            fold_sizes == smallest_fold_size
        )

        selected_fold = int(
            random_generator.choice(
                candidate_folds
            )
        )

        group_to_fold[
            row.visual_group
        ] = selected_fold

        fold_sizes[selected_fold] += int(
            row.patch_count
        )

    fold_assignments = (
        patch_metadata["visual_group"]
        .map(group_to_fold)
        .to_numpy(dtype=np.int64)
    )

    if (fold_assignments < 0).any():
        raise RuntimeError(
            "Neki patch-evi nisu dobili fold."
        )

    split_values = np.full(
        len(patch_metadata),
        fill_value="train",
        dtype="<U10",
    )

    split_values[
        fold_assignments == VALIDATION_FOLD
    ] = "validation"

    split_values[
        fold_assignments == TEST_FOLD
    ] = "test"

    result = patch_metadata.copy()
    result["split"] = split_values

    print(
        "Veličine svih foldova:",
        fold_sizes.tolist(),
    )

    return result


def add_patch_record(
    records,
    patch_id,
    pair_id,
    image_id,
    image_path,
    target_class,
    source_defect_class,
    box,
):
    xmin, ymin, xmax, ymax = box

    records.append(
        {
            "patch_id": patch_id,
            "pair_id": pair_id,
            "image_id": image_id,
            "image_path": str(image_path),
            "target_class": target_class,
            "source_defect_class": (
                source_defect_class
            ),
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "patch_width": xmax - xmin,
            "patch_height": ymax - ymin,
        }
    )


def main():
    annotations_directory = (
        DATASET_DIRECTORY / "Annotations"
    )
    images_directory = (
        DATASET_DIRECTORY / "JPEGImages"
    )

    xml_paths = sorted(
        annotations_directory.glob("*.xml")
    )

    random_generator = np.random.default_rng(
        RANDOM_STATE
    )

    image_records = []
    annotations_by_image = {}

    invalid_box_count = 0
    empty_annotation_count = 0

    print("Čitanje anotacija...")

    for xml_path in xml_paths:
        annotation = read_annotation(xml_path)

        image_id = xml_path.stem
        image_path = (
            images_directory / f"{image_id}.jpg"
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Nedostaje slika: {image_path}"
            )

        boxes = annotation["boxes"]

        if not boxes:
            empty_annotation_count += 1
            continue

        annotations_by_image[image_id] = annotation

        image_records.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
            }
        )

    image_metadata = pd.DataFrame(
        image_records
    )

    print(
        "Slike sa najmanje jednim defektom:",
        len(image_metadata),
    )
    print(
        "Isključene prazne anotacije:",
        empty_annotation_count,
    )

    print("Računanje perceptual hash vrednosti...")

    image_metadata = add_perceptual_hashes(
        image_metadata
    )

    print("Formiranje vizuelnih grupa...")

    image_metadata = assign_visual_groups(
        image_metadata,
        max_hamming_distance=6,
    )

    visual_group_by_image = (
        image_metadata
        .set_index("image_id")["visual_group"]
        .to_dict()
    )

    print("Generisanje patch parova...")

    patch_records = []

    skipped_positive_boxes = 0
    generated_pair_count = 0

    for image_id, annotation in (
        annotations_by_image.items()
    ):
        image_width = annotation["width"]
        image_height = annotation["height"]
        boxes = annotation["boxes"]

        number_to_select = min(
            MAX_PAIRS_PER_IMAGE,
            len(boxes),
        )

        selected_indices = (
            random_generator.choice(
                len(boxes),
                size=number_to_select,
                replace=False,
            )
        )

        forbidden_boxes = [
            expand_box(
                annotation_box["box"],
                image_width,
                image_height,
                NEGATIVE_EXCLUSION_RATIO,
            )
            for annotation_box in boxes
        ]

        image_path = (
            images_directory / f"{image_id}.jpg"
        )

        for local_index, box_index in enumerate(
            sorted(selected_indices.tolist())
        ):
            annotation_box = boxes[box_index]

            positive_box = make_positive_patch_box(
                box=annotation_box["box"],
                image_width=image_width,
                image_height=image_height,
            )

            (
                positive_xmin,
                positive_ymin,
                positive_xmax,
                positive_ymax,
            ) = positive_box

            crop_width = (
                positive_xmax - positive_xmin
            )
            crop_height = (
                positive_ymax - positive_ymin
            )

            negative_box = sample_negative_box(
                crop_width=crop_width,
                crop_height=crop_height,
                image_width=image_width,
                image_height=image_height,
                forbidden_boxes=forbidden_boxes,
                reference_box=positive_box,
                random_generator=random_generator,
            )

            if negative_box is None:
                skipped_positive_boxes += 1
                continue

            pair_id = (
                f"{image_id}_pair_{local_index:02d}"
            )

            add_patch_record(
                records=patch_records,
                patch_id=f"{pair_id}_defect",
                pair_id=pair_id,
                image_id=image_id,
                image_path=image_path,
                target_class="defect",
                source_defect_class=(
                    annotation_box["class_name"]
                ),
                box=positive_box,
            )

            add_patch_record(
                records=patch_records,
                patch_id=f"{pair_id}_no_defect",
                pair_id=pair_id,
                image_id=image_id,
                image_path=image_path,
                target_class="no_defect",
                source_defect_class=(
                    annotation_box["class_name"]
                ),
                box=negative_box,
            )

            generated_pair_count += 1

    patch_metadata = pd.DataFrame(
        patch_records
    )

    patch_metadata["visual_group"] = (
        patch_metadata["image_id"].map(
            visual_group_by_image
        )
    )

    if patch_metadata[
        "visual_group"
    ].isna().any():
        raise RuntimeError(
            "Neki patch-evi nemaju vizuelnu grupu."
        )

    patch_metadata = assign_patch_splits(
        patch_metadata
    )

    group_split_counts = (
        patch_metadata
        .groupby("visual_group")["split"]
        .nunique()
    )

    image_split_counts = (
        patch_metadata
        .groupby("image_id")["split"]
        .nunique()
    )

    pair_label_counts = (
        patch_metadata
        .groupby("pair_id")["target_class"]
        .nunique()
    )

    if group_split_counts.max() != 1:
        raise RuntimeError(
            "Vizuelna grupa se pojavljuje "
            "u više splitova."
        )

    if image_split_counts.max() != 1:
        raise RuntimeError(
            "Slika se pojavljuje u više splitova."
        )

    if not (pair_label_counts == 2).all():
        raise RuntimeError(
            "Neki par nema oba binarna patch-a."
        )

    patch_metadata = patch_metadata.sort_values(
        [
            "split",
            "visual_group",
            "image_id",
            "pair_id",
            "target_class",
        ]
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    patch_metadata.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    report = {
        "experiment": (
            "experiment_1_patch_binary"
        ),
        "configuration": {
            "max_pairs_per_image": (
                MAX_PAIRS_PER_IMAGE
            ),
            "positive_context_ratio": (
                POSITIVE_CONTEXT_RATIO
            ),
            "minimum_patch_side": (
                MIN_PATCH_SIDE
            ),
            "maximum_patch_side": (
                MAX_PATCH_SIDE
            ),
            "negative_exclusion_ratio": (
                NEGATIVE_EXCLUSION_RATIO
            ),
            "maximum_negative_attempts": (
                MAX_NEGATIVE_ATTEMPTS
            ),
            "perceptual_hash_threshold": 6,
            "random_state": RANDOM_STATE,
        },
        "counts": {
            "images_with_annotations": len(
                image_metadata
            ),
            "excluded_empty_annotations": (
                empty_annotation_count
            ),
            "visual_groups": int(
                image_metadata[
                    "visual_group"
                ].nunique()
            ),
            "generated_pairs": (
                generated_pair_count
            ),
            "generated_patches": len(
                patch_metadata
            ),
            "skipped_positive_boxes": (
                skipped_positive_boxes
            ),
        },
        "patches_by_target": {
            str(key): int(value)
            for key, value in (
                patch_metadata[
                    "target_class"
                ].value_counts().items()
            )
        },
        "patches_by_split": {
            str(key): int(value)
            for key, value in (
                patch_metadata[
                    "split"
                ].value_counts().items()
            )
        },
        "patches_by_target_and_split": (
            pd.crosstab(
                patch_metadata["target_class"],
                patch_metadata["split"],
            ).to_dict()
        ),
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print("\nPatch metadata je napravljen.")
    print(
        "Broj vizuelnih grupa:",
        image_metadata[
            "visual_group"
        ].nunique(),
    )
    print(
        "Broj generisanih parova:",
        generated_pair_count,
    )
    print(
        "Broj generisanih patch-eva:",
        len(patch_metadata),
    )
    print(
        "Preskočeni pozitivni box-evi:",
        skipped_positive_boxes,
    )

    print("\nPatch-evi po klasi:")
    print(
        patch_metadata[
            "target_class"
        ].value_counts()
    )

    print("\nPatch-evi po splitu:")
    print(
        patch_metadata["split"].value_counts()
    )

    print("\nKlase po splitu:")
    print(
        pd.crosstab(
            patch_metadata["target_class"],
            patch_metadata["split"],
        )
    )

    print(
        "\nNajveći broj splitova po "
        "visual_group:",
        group_split_counts.max(),
    )
    print(
        "Najveći broj splitova po image_id:",
        image_split_counts.max(),
    )

    print("\nSačuvano:")
    print(OUTPUT_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()