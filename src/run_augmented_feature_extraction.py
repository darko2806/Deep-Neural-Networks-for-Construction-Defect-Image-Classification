import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset

from src.feature_extraction import (
    create_feature_extractor,
    extract_features,
)


FEATURE_EXTRACTOR = "inception_resnet_v2"
AUGMENTED_CLASSES = [
    "bulge",
    "corrosion",
    "leakage",
]
RANDOM_SEED = 42

METADATA_PATH = Path("data/metadata.csv")

OUTPUT_DIRECTORY = Path(
    "data/results/experiment_2b_augmentation"
)
AUGMENTED_METADATA_PATH = (
    OUTPUT_DIRECTORY / "augmentation_metadata.csv"
)
REPORT_PATH = (
    OUTPUT_DIRECTORY / "augmentation_report.json"
)
FEATURE_PATH = Path(
    "data/features/experiment_2b_augmentation/"
    "inception_resnet_v2.npz"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Train-only augmentacija i ekstrakcija "
            "Inception-ResNet-v2 feature-a za Eksperiment 2B"
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help=(
            "Ciljni broj train primera po augmentiranoj klasi. "
            "Podrazumevano se koristi broj druge najveće klase."
        ),
    )

    return parser.parse_args()


def deterministic_parameters(
    source_image_id,
    augmentation_index,
):
    key = (
        f"{RANDOM_SEED}|{source_image_id}|"
        f"{augmentation_index}"
    )

    digest = hashlib.sha256(
        key.encode("utf-8")
    ).digest()

    local_seed = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    generator = random.Random(local_seed)

    return {
        "horizontal_flip": generator.random() < 0.5,
        "rotation_degrees": generator.uniform(
            -10.0,
            10.0,
        ),
        "brightness_factor": generator.uniform(
            0.90,
            1.10,
        ),
        "contrast_factor": generator.uniform(
            0.90,
            1.10,
        ),
        "saturation_factor": generator.uniform(
            0.90,
            1.10,
        ),
    }


def create_augmented_metadata(
    metadata,
    target_count=None,
):
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
            "Nedostaju kolone: "
            f"{sorted(missing_columns)}"
        )

    original = metadata.copy()

    original["source_image_id"] = (
        original["image_id"].astype(str)
    )
    original["is_augmented"] = False
    original["augmentation_index"] = 0
    original["horizontal_flip"] = False
    original["rotation_degrees"] = 0.0
    original["brightness_factor"] = 1.0
    original["contrast_factor"] = 1.0
    original["saturation_factor"] = 1.0

    train_metadata = original.loc[
        original["split"].eq("train")
    ].copy()

    train_counts = (
        train_metadata["target_class"]
        .value_counts()
        .sort_values(ascending=False)
    )

    if target_count is None:
        if len(train_counts) < 2:
            raise ValueError(
                "Potrebne su najmanje dve klase"
            )

        target_count = int(train_counts.iloc[1])

    augmented_records = []

    for class_name in AUGMENTED_CLASSES:
        class_sources = train_metadata.loc[
            train_metadata["target_class"].eq(
                class_name
            )
        ].copy()

        if class_sources.empty:
            raise ValueError(
                f"Klasa {class_name} nema train primere"
            )

        current_count = len(class_sources)
        required_count = max(
            0,
            target_count - current_count,
        )

        source_records = (
            class_sources
            .sample(
                frac=1.0,
                random_state=RANDOM_SEED,
            )
            .to_dict("records")
        )

        source_usage = {
            str(record["image_id"]): 0
            for record in source_records
        }

        for augmentation_number in range(
            required_count
        ):
            source = source_records[
                augmentation_number
                % len(source_records)
            ].copy()

            source_image_id = str(
                source["image_id"]
            )

            source_usage[source_image_id] += 1

            augmentation_index = source_usage[
                source_image_id
            ]

            parameters = deterministic_parameters(
                source_image_id,
                augmentation_index,
            )

            sample_id = (
                f"{source_image_id}"
                f"__aug_{augmentation_index:02d}"
            )

            source["image_id"] = sample_id
            source["source_image_id"] = (
                source_image_id
            )
            source["is_augmented"] = True
            source["augmentation_index"] = (
                augmentation_index
            )
            source.update(parameters)

            augmented_records.append(source)

    augmented = pd.DataFrame(
        augmented_records,
        columns=original.columns,
    )

    complete_metadata = pd.concat(
        [
            original,
            augmented,
        ],
        ignore_index=True,
    )

    if complete_metadata["image_id"].duplicated().any():
        duplicates = complete_metadata.loc[
            complete_metadata["image_id"].duplicated(
                keep=False
            ),
            "image_id",
        ].tolist()

        raise ValueError(
            "Duplirani sample ID-jevi: "
            f"{duplicates[:10]}"
        )

    augmented_rows = complete_metadata.loc[
        complete_metadata["is_augmented"]
    ]

    if not augmented_rows["split"].eq("train").all():
        raise ValueError(
            "Augmentirani primeri moraju pripadati "
            "isključivo train skupu"
        )

    original_counts_after = (
        complete_metadata.loc[
            ~complete_metadata["is_augmented"]
        ]["split"]
        .value_counts()
        .sort_index()
    )

    original_counts_before = (
        metadata["split"]
        .value_counts()
        .sort_index()
    )

    if not original_counts_after.equals(
        original_counts_before
    ):
        raise ValueError(
            "Originalni splitovi su promenjeni"
        )

    split_count_per_group = (
        complete_metadata.groupby(
            "visual_group"
        )["split"]
        .nunique()
    )

    if split_count_per_group.max() != 1:
        raise ValueError(
            "Otkriveno je curenje visual_group između splitova"
        )

    source_split_count = (
        complete_metadata.groupby(
            "source_image_id"
        )["split"]
        .nunique()
    )

    if source_split_count.max() != 1:
        raise ValueError(
            "Izvorna slika se pojavljuje u više splitova"
        )

    return complete_metadata, target_count


def augment_image(image, row):
    if bool(row["horizontal_flip"]):
        image = ImageOps.mirror(image)

    image = image.rotate(
        angle=float(row["rotation_degrees"]),
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(124, 116, 104),
    )

    image = ImageEnhance.Brightness(
        image
    ).enhance(
        float(row["brightness_factor"])
    )

    image = ImageEnhance.Contrast(
        image
    ).enhance(
        float(row["contrast_factor"])
    )

    image = ImageEnhance.Color(
        image
    ).enhance(
        float(row["saturation_factor"])
    )

    return image


class AugmentedFeatureDataset(Dataset):
    def __init__(
        self,
        metadata,
        inference_transform,
    ):
        self.metadata = (
            metadata
            .reset_index(drop=True)
            .copy()
        )
        self.inference_transform = (
            inference_transform
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]

        with Image.open(
            Path(row["image_path"])
        ) as image:
            image = image.convert("RGB")

            if bool(row["is_augmented"]):
                image = augment_image(
                    image,
                    row,
                )

            image = self.inference_transform(
                image
            )

        return {
            "image": image,
            "target_class": str(
                row["target_class"]
            ),
            "image_id": str(row["image_id"]),
        }


def integer_dictionary(series):
    return {
        str(key): int(value)
        for key, value in series.items()
    }


def main():
    args = parse_arguments()

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Nedostaje: {METADATA_PATH}"
        )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    augmented_metadata, target_count = (
        create_augmented_metadata(
            metadata,
            target_count=args.target_count,
        )
    )

    original_train_counts = (
        metadata.loc[
            metadata["split"].eq("train"),
            "target_class",
        ]
        .value_counts()
        .sort_index()
    )

    final_train_counts = (
        augmented_metadata.loc[
            augmented_metadata["split"].eq(
                "train"
            ),
            "target_class",
        ]
        .value_counts()
        .sort_index()
    )

    augmentation_counts = (
        augmented_metadata.loc[
            augmented_metadata[
                "is_augmented"
            ],
            "target_class",
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "Eksperiment 2B – train-only augmentacija"
    )
    print("Feature extractor:", FEATURE_EXTRACTOR)
    print("Ciljni broj:", target_count)

    print("\nOriginalni train broj po klasama:")
    print(original_train_counts)

    print("\nBroj novih augmentiranih primera:")
    print(augmentation_counts)

    print("\nKonačni train broj po klasama:")
    print(final_train_counts)

    print(
        "\nOriginalnih slika:",
        len(metadata),
    )
    print(
        "Augmentiranih prikaza:",
        int(
            augmented_metadata[
                "is_augmented"
            ].sum()
        ),
    )
    print(
        "Ukupno redova:",
        len(augmented_metadata),
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    FEATURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    augmented_metadata.to_csv(
        AUGMENTED_METADATA_PATH,
        index=False,
    )

    model, inference_transform, device = (
        create_feature_extractor(
            FEATURE_EXTRACTOR
        )
    )

    print("\nUređaj:", device)
    print("Batch size:", args.batch_size)

    dataset = AugmentedFeatureDataset(
        metadata=augmented_metadata,
        inference_transform=inference_transform,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    result = extract_features(
        model=model,
        data_loader=data_loader,
        device=device,
    )

    expected_ids = (
        augmented_metadata["image_id"]
        .astype(str)
        .to_numpy()
    )

    if not np.array_equal(
        result["image_ids"].astype(str),
        expected_ids,
    ):
        raise ValueError(
            "Redosled feature-a ne odgovara metadata tabeli"
        )

    if not np.isfinite(
        result["features"]
    ).all():
        raise ValueError(
            "Feature matrica sadrži NaN ili Inf"
        )

    np.savez(
        FEATURE_PATH,
        features=result["features"].astype(
            np.float32
        ),
        image_ids=result[
            "image_ids"
        ].astype(str),
        target_classes=result[
            "target_classes"
        ].astype(str),
        splits=augmented_metadata[
            "split"
        ].to_numpy(dtype=str),
    )

    report = {
        "experiment": (
            "experiment_2b_train_only_augmentation"
        ),
        "feature_extractor": FEATURE_EXTRACTOR,
        "random_seed": RANDOM_SEED,
        "augmented_classes": AUGMENTED_CLASSES,
        "target_train_count": int(
            target_count
        ),
        "augmentation": {
            "horizontal_flip_probability": 0.5,
            "rotation_degrees": [-10.0, 10.0],
            "brightness_factor": [0.90, 1.10],
            "contrast_factor": [0.90, 1.10],
            "saturation_factor": [0.90, 1.10],
        },
        "original_train_counts": (
            integer_dictionary(
                original_train_counts
            )
        ),
        "augmentation_counts": (
            integer_dictionary(
                augmentation_counts
            )
        ),
        "final_train_counts": (
            integer_dictionary(
                final_train_counts
            )
        ),
        "sample_counts": {
            "original_total": int(
                len(metadata)
            ),
            "augmented_train_views": int(
                augmented_metadata[
                    "is_augmented"
                ].sum()
            ),
            "combined_total": int(
                len(augmented_metadata)
            ),
        },
        "leakage_checks": {
            "augmentations_only_in_train": True,
            "original_splits_unchanged": True,
            "one_split_per_visual_group": True,
            "one_split_per_source_image": True,
        },
        "metadata_path": str(
            AUGMENTED_METADATA_PATH
        ),
        "feature_path": str(FEATURE_PATH),
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with np.load(
        FEATURE_PATH,
        allow_pickle=False,
    ) as saved:
        saved_shape = saved[
            "features"
        ].shape

    print("\nEkstrakcija je završena.")
    print("Metadata:", AUGMENTED_METADATA_PATH)
    print("Izveštaj:", REPORT_PATH)
    print("Feature-i:", FEATURE_PATH)
    print("Oblik feature matrice:", saved_shape)
    print(
        "Provera sa allow_pickle=False: uspešna"
    )


if __name__ == "__main__":
    main()