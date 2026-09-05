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
from src.image_dataset import DefectImageDataset


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Izvlačenje CNN feature vektora"
    )

    parser.add_argument(
        "--model",
        choices=sorted(FEATURE_EXTRACTOR_MODELS),
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


def main():
    args = parse_arguments()

    metadata_path = Path("data/metadata.csv")
    output_path = Path(
        f"data/features/{args.model}.npz"
    )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata fajl ne postoji: {metadata_path}"
        )

    metadata = pd.read_csv(metadata_path)

    print(f"Model: {args.model}")
    print(f"Broj slika: {len(metadata)}")
    print(f"Batch size: {args.batch_size}")

    model, transform, device = create_feature_extractor(
        args.model
    )

    print(f"Uređaj: {device}")

    dataset = DefectImageDataset(
        metadata=metadata,
        transform=transform,
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

    expected_image_ids = metadata["image_id"].to_numpy()

    if not np.array_equal(
        result["image_ids"],
        expected_image_ids,
    ):
        raise ValueError(
            "Redosled feature vektora ne odgovara metadata redovima"
        )

    if not np.isfinite(result["features"]).all():
        raise ValueError(
            "Feature matrica sadrži NaN ili beskonačne vrednosti"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        output_path,
        features=result["features"],
        image_ids=result["image_ids"].astype(str),
        target_classes=result["target_classes"].astype(str),
        splits=metadata["split"].to_numpy(dtype=str),
    )

    print(f"\nFeature-i su sačuvani u: {output_path}")
    print(f"Oblik feature matrice: {result['features'].shape}")


if __name__ == "__main__":
    main()