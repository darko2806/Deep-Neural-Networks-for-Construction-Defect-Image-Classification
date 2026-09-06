from pathlib import Path

import numpy as np


REQUIRED_ARRAYS = {
    "features",
    "image_ids",
    "target_classes",
    "splits",
}

EXPECTED_SPLITS = {
    "train",
    "validation",
    "test",
}


def load_feature_splits(feature_path):
    feature_path = Path(feature_path)

    if not feature_path.is_file():
        raise FileNotFoundError(
            f"Feature fajl ne postoji: {feature_path}"
        )

    with np.load(feature_path, allow_pickle=False) as data:
        missing_arrays = REQUIRED_ARRAYS - set(data.files)

        if missing_arrays:
            raise ValueError(
                f"Nedostaju nizovi: {sorted(missing_arrays)}"
            )

        features = data["features"]
        image_ids = data["image_ids"]
        target_classes = data["target_classes"]
        splits = data["splits"]

    if features.ndim != 2:
        raise ValueError(
            "Feature matrica mora imati oblik "
            "[broj_slika, broj_feature-a]"
        )

    n_samples = len(features)

    arrays = {
        "image_ids": image_ids,
        "target_classes": target_classes,
        "splits": splits,
    }

    for array_name, array in arrays.items():
        if len(array) != n_samples:
            raise ValueError(
                f"{array_name} nema isti broj redova kao features"
            )

    if not np.isfinite(features).all():
        raise ValueError(
            "Feature matrica sadrži NaN ili beskonačne vrednosti"
        )

    found_splits = set(np.unique(splits))

    if found_splits != EXPECTED_SPLITS:
        raise ValueError(
            f"Očekivani splitovi su {sorted(EXPECTED_SPLITS)}, "
            f"a pronađeni su {sorted(found_splits)}"
        )

    feature_splits = {}

    for split_name in ("train", "validation", "test"):
        split_mask = splits == split_name

        feature_splits[split_name] = {
            "X": features[split_mask],
            "y": target_classes[split_mask],
            "image_ids": image_ids[split_mask],
        }

    return feature_splits