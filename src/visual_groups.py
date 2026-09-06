from collections import defaultdict
import numpy as np
from pathlib import Path

import imagehash
from PIL import Image


def compute_perceptual_hash(image_path, hash_size=8):
    image_path = Path(image_path)

    with Image.open(image_path) as image:
        perceptual_hash = imagehash.phash(
            image.convert("RGB"),
            hash_size=hash_size,
        )

    return str(perceptual_hash)


def add_perceptual_hashes(metadata):
    if "image_path" not in metadata.columns:
        raise ValueError(
            "Nedostaje obavezna kolona: image_path"
        )

    metadata_with_hashes = metadata.copy()

    metadata_with_hashes["perceptual_hash"] = (
        metadata_with_hashes["image_path"]
        .apply(compute_perceptual_hash)
    )

    return metadata_with_hashes


def assign_visual_groups(
    metadata,
    max_hamming_distance=6,
    block_size=256,
):
    if "perceptual_hash" not in metadata.columns:
        raise ValueError(
            "Nedostaje obavezna kolona: perceptual_hash"
        )

    if not 0 <= max_hamming_distance <= 64:
        raise ValueError(
            "Hamming udaljenost mora biti između 0 i 64"
        )

    if block_size < 1:
        raise ValueError(
            "block_size mora biti pozitivan"
        )

    metadata_with_groups = metadata.copy()

    hash_values = np.asarray(
        [
            int(hash_value, 16)
            for hash_value in metadata_with_groups[
                "perceptual_hash"
            ]
        ],
        dtype=np.uint64,
    )

    n_images = len(metadata_with_groups)
    parent = list(range(n_images))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]

        return index

    def union(first_index, second_index):
        first_root = find(first_index)
        second_root = find(second_index)

        if first_root != second_root:
            parent[second_root] = first_root

    for start in range(0, n_images, block_size):
        end = min(
            start + block_size,
            n_images,
        )

        hash_block = hash_values[start:end]
        remaining_hashes = hash_values[start:]

        distances = np.bitwise_count(
            np.bitwise_xor(
                hash_block[:, None],
                remaining_hashes[None, :],
            )
        )

        row_indices, column_indices = np.where(
            distances <= max_hamming_distance
        )

        first_indices = start + row_indices
        second_indices = start + column_indices

        valid_pairs = (
            second_indices > first_indices
        )

        for first_index, second_index in zip(
            first_indices[valid_pairs],
            second_indices[valid_pairs],
        ):
            union(
                int(first_index),
                int(second_index),
            )

    indices_by_root = defaultdict(list)

    for index in range(n_images):
        indices_by_root[find(index)].append(index)

    sorted_roots = sorted(
        indices_by_root,
        key=lambda root: min(
            metadata_with_groups.iloc[index][
                "image_id"
            ]
            for index in indices_by_root[root]
        ),
    )

    group_name_by_root = {
        root: f"visual_group_{group_number:05d}"
        for group_number, root in enumerate(sorted_roots)
    }

    metadata_with_groups["visual_group"] = [
        group_name_by_root[find(index)]
        for index in range(n_images)
    ]

    return metadata_with_groups

def find_visual_group_class_conflicts(metadata):
    required_columns = {
        "visual_group",
        "target_class",
    }

    missing_columns = (
        required_columns - set(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Nedostaju kolone: {sorted(missing_columns)}"
        )

    class_counts = (
        metadata.groupby("visual_group")[
            "target_class"
        ]
        .nunique()
    )

    conflicting_groups = class_counts[
        class_counts > 1
    ].index

    conflicts = metadata.loc[
        metadata["visual_group"].isin(
            conflicting_groups
        ),
        [
            "image_id",
            "image_path",
            "target_class",
            "perceptual_hash",
            "visual_group",
        ],
    ].copy()

    return conflicts.sort_values(
        ["visual_group", "image_id"]
    )

def remove_visual_group_class_conflicts(metadata):
    conflicts = find_visual_group_class_conflicts(
        metadata
    )

    conflicting_groups = set(
        conflicts["visual_group"]
    )

    cleaned_metadata = metadata.loc[
        ~metadata["visual_group"].isin(
            conflicting_groups
        )
    ].copy()

    cleaned_metadata = cleaned_metadata.reset_index(
        drop=True
    )

    return cleaned_metadata