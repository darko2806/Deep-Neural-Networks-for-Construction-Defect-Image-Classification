from pathlib import Path

import numpy as np
import pandas as pd


ORIGINAL_FEATURE_PATH = Path(
    "data/features/experiment_2_multiclass/"
    "inception_resnet_v2.npz"
)
EXTRACTED_AUGMENTED_FEATURE_PATH = Path(
    "data/features/experiment_2b_augmentation/"
    "inception_resnet_v2.npz"
)
AUGMENTATION_METADATA_PATH = Path(
    "data/results/experiment_2b_augmentation/"
    "augmentation_metadata.csv"
)
OUTPUT_PATH = Path(
    "data/features/experiment_2b_augmentation/"
    "inception_resnet_v2_merged.npz"
)


def load_feature_file(path):
    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        return {
            key: data[key]
            for key in data.files
        }


def parse_boolean(series):
    if series.dtype == bool:
        return series

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    values = {
        "true": True,
        "false": False,
    }

    if not normalized.isin(values).all():
        raise ValueError(
            "is_augmented sadrži nepoznate vrednosti"
        )

    return normalized.map(values).astype(bool)


def create_index(image_ids):
    index = {}

    for position, image_id in enumerate(
        image_ids.astype(str)
    ):
        if image_id in index:
            raise ValueError(
                f"Dupliran image_id: {image_id}"
            )

        index[image_id] = position

    return index


def main():
    for path in (
        ORIGINAL_FEATURE_PATH,
        EXTRACTED_AUGMENTED_FEATURE_PATH,
        AUGMENTATION_METADATA_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Nedostaje: {path}"
            )

    original = load_feature_file(
        ORIGINAL_FEATURE_PATH
    )
    extracted = load_feature_file(
        EXTRACTED_AUGMENTED_FEATURE_PATH
    )
    metadata = pd.read_csv(
        AUGMENTATION_METADATA_PATH
    )

    metadata["is_augmented"] = parse_boolean(
        metadata["is_augmented"]
    )

    original_index = create_index(
        original["image_ids"]
    )
    extracted_index = create_index(
        extracted["image_ids"]
    )

    feature_dimension = original[
        "features"
    ].shape[1]

    if (
        extracted["features"].shape[1]
        != feature_dimension
    ):
        raise ValueError(
            "Feature dimenzije nisu jednake"
        )

    merged_features = np.empty(
        (
            len(metadata),
            feature_dimension,
        ),
        dtype=np.float32,
    )

    original_rows = 0
    augmented_rows = 0

    for row_index, row in metadata.iterrows():
        sample_id = str(row["image_id"])

        if row["is_augmented"]:
            if sample_id not in extracted_index:
                raise ValueError(
                    "Nedostaje augmentirani feature: "
                    f"{sample_id}"
                )

            source_position = extracted_index[
                sample_id
            ]
            merged_features[row_index] = (
                extracted["features"][
                    source_position
                ]
            )
            augmented_rows += 1
        else:
            if sample_id not in original_index:
                raise ValueError(
                    "Nedostaje originalni feature: "
                    f"{sample_id}"
                )

            source_position = original_index[
                sample_id
            ]
            merged_features[row_index] = (
                original["features"][
                    source_position
                ]
            )
            original_rows += 1

    target_classes = metadata[
        "target_class"
    ].to_numpy(dtype=str)

    splits = metadata[
        "split"
    ].to_numpy(dtype=str)

    image_ids = metadata[
        "image_id"
    ].to_numpy(dtype=str)

    if not np.isfinite(
        merged_features
    ).all():
        raise ValueError(
            "Spojena feature matrica sadrži NaN ili Inf"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        OUTPUT_PATH,
        features=merged_features,
        image_ids=image_ids,
        target_classes=target_classes,
        splits=splits,
    )

    for split_name in ("validation", "test"):
        merged_mask = splits == split_name
        original_mask = (
            original["splits"].astype(str)
            == split_name
        )

        merged_ids = image_ids[merged_mask]
        original_ids = original[
            "image_ids"
        ][original_mask].astype(str)

        if not np.array_equal(
            merged_ids,
            original_ids,
        ):
            raise ValueError(
                f"{split_name} ID-jevi nisu identični"
            )

        merged_split_features = (
            merged_features[merged_mask]
        )
        original_split_features = (
            original["features"][
                original_mask
            ]
        )

        if not np.array_equal(
            merged_split_features,
            original_split_features,
        ):
            raise ValueError(
                f"{split_name} feature-i nisu identični"
            )

    print("Spajanje feature-a je završeno.")
    print(
        "Originalnih feature redova:",
        original_rows,
    )
    print(
        "Augmentiranih feature redova:",
        augmented_rows,
    )
    print(
        "Oblik spojene matrice:",
        merged_features.shape,
    )
    print(
        "Validation i test su identični "
        "originalnom Eksperimentu 2."
    )
    print("Sačuvano:", OUTPUT_PATH)


if __name__ == "__main__":
    main()