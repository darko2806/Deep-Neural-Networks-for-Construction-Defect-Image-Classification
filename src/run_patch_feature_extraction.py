import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.feature_extraction import (
    FEATURE_EXTRACTOR_MODELS,
    create_feature_extractor,
    extract_features,
)
from src.patch_dataset import (
    PatchDefectDataset,
)


METADATA_PATH = Path(
    "data/patch_metadata.csv"
)

OUTPUT_DIRECTORY = Path(
    "data/features/experiment_1_binary"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Ekstrakcija CNN feature-a za "
            "patch-level binarnu klasifikaciju."
        )
    )

    parser.add_argument(
        "--model",
        choices=sorted(
            FEATURE_EXTRACTOR_MODELS
        ),
        required=True,
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

    return parser.parse_args()


def validate_metadata(metadata):
    required_columns = {
        "patch_id",
        "pair_id",
        "image_id",
        "image_path",
        "target_class",
        "visual_group",
        "split",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    }

    missing_columns = (
        required_columns
        - set(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            "Nedostaju kolone: "
            f"{sorted(missing_columns)}"
        )

    if metadata["patch_id"].duplicated().any():
        raise ValueError(
            "patch_id vrednosti nisu jedinstvene."
        )

    expected_classes = {
        "defect",
        "no_defect",
    }

    found_classes = set(
        metadata[
            "target_class"
        ].unique()
    )

    if found_classes != expected_classes:
        raise ValueError(
            "Neočekivane klase: "
            f"{sorted(found_classes)}"
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
            "Neočekivani splitovi: "
            f"{sorted(found_splits)}"
        )

    invalid_widths = (
        metadata["xmax"]
        <= metadata["xmin"]
    )

    invalid_heights = (
        metadata["ymax"]
        <= metadata["ymin"]
    )

    if (
        invalid_widths.any()
        or invalid_heights.any()
    ):
        raise ValueError(
            "Pronađene su neispravne "
            "patch koordinate."
        )

    group_split_counts = (
        metadata
        .groupby("visual_group")["split"]
        .nunique()
    )

    image_split_counts = (
        metadata
        .groupby("image_id")["split"]
        .nunique()
    )

    if group_split_counts.max() != 1:
        raise ValueError(
            "Vizuelna grupa prelazi između "
            "splitova."
        )

    if image_split_counts.max() != 1:
        raise ValueError(
            "Originalna slika prelazi između "
            "splitova."
        )


def main():
    arguments = parse_arguments()

    if arguments.batch_size < 1:
        raise ValueError(
            "--batch-size mora biti pozitivan."
        )

    if arguments.num_workers < 0:
        raise ValueError(
            "--num-workers ne može biti "
            "negativan."
        )

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            "Patch metadata ne postoji: "
            f"{METADATA_PATH}"
        )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    validate_metadata(metadata)

    print(
        "Feature extractor:",
        arguments.model,
    )
    print(
        "Broj patch-eva:",
        len(metadata),
    )
    print(
        "Batch size:",
        arguments.batch_size,
    )

    print("\nPatch-evi po splitu:")
    print(
        metadata[
            "split"
        ].value_counts()
    )

    print("\nKlase po splitu:")
    print(
        pd.crosstab(
            metadata["target_class"],
            metadata["split"],
        )
    )

    model, transform, device = (
        create_feature_extractor(
            arguments.model
        )
    )

    print("\nUređaj:", device)

    dataset = PatchDefectDataset(
        metadata=metadata,
        transform=transform,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            arguments.num_workers > 0
        ),
    )

    result = extract_features(
        model=model,
        data_loader=data_loader,
        device=device,
    )

    expected_patch_ids = (
        metadata["patch_id"]
        .to_numpy(dtype=str)
    )

    if not np.array_equal(
        result["image_ids"],
        expected_patch_ids,
    ):
        raise ValueError(
            "Redosled feature vektora ne "
            "odgovara redosledu patch-eva."
        )

    if not np.isfinite(
        result["features"]
    ).all():
        raise ValueError(
            "Feature matrica sadrži NaN ili "
            "beskonačne vrednosti."
        )

    if len(result["features"]) != len(metadata):
        raise ValueError(
            "Broj feature vektora ne odgovara "
            "broju patch-eva."
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIRECTORY / (
        f"{arguments.model}.npz"
    )

    np.savez_compressed(
        output_path,
        features=(
            result["features"]
        ),
        image_ids=expected_patch_ids,
        patch_ids=expected_patch_ids,
        source_image_ids=(
            metadata["image_id"]
            .to_numpy(dtype=str)
        ),
        pair_ids=(
            metadata["pair_id"]
            .to_numpy(dtype=str)
        ),
        visual_groups=(
            metadata["visual_group"]
            .to_numpy(dtype=str)
        ),
        target_classes=(
            metadata["target_class"]
            .to_numpy(dtype=str)
        ),
        splits=(
            metadata["split"]
            .to_numpy(dtype=str)
        ),
    )

    print("\nEkstrakcija je završena.")
    print("Sačuvano:", output_path)
    print(
        "Oblik feature matrice:",
        result["features"].shape,
    )
    print(
        "Feature dtype:",
        result["features"].dtype,
    )

    with np.load(
        output_path,
        allow_pickle=False,
    ) as saved_data:
        if not np.array_equal(
            saved_data["patch_ids"],
            expected_patch_ids,
        ):
            raise ValueError(
                "Provera sačuvanog NPZ fajla "
                "nije uspela."
            )

        if not np.isfinite(
            saved_data["features"]
        ).all():
            raise ValueError(
                "Sačuvani feature-i nisu "
                "konačni brojevi."
            )

    print(
        "Provera sa allow_pickle=False: uspešna"
    )


if __name__ == "__main__":
    main()