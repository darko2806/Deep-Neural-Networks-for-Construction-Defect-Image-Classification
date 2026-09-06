import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


METADATA_PATH = Path("data/patch_metadata.csv")

OUTPUT_DIRECTORY = Path(
    "data/results/experiment_1_binary"
)

EXPECTED_CLASSES = {
    "defect",
    "no_defect",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Vizuelna provera defect/no_defect "
            "patch parova."
        )
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "validation",
            "test",
        ],
        default="train",
    )

    parser.add_argument(
        "--pairs-per-class",
        type=int,
        default=2,
        help=(
            "Broj parova za svaki izvorni "
            "tip defekta."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def validate_metadata(metadata):
    required_columns = {
        "patch_id",
        "pair_id",
        "image_id",
        "image_path",
        "target_class",
        "source_defect_class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
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

    invalid_classes = (
        set(metadata["target_class"].unique())
        - EXPECTED_CLASSES
    )

    if invalid_classes:
        raise ValueError(
            "Neočekivane target klase: "
            f"{sorted(invalid_classes)}"
        )

    pair_sizes = metadata.groupby(
        "pair_id"
    ).size()

    invalid_pair_sizes = pair_sizes[
        pair_sizes != 2
    ]

    if not invalid_pair_sizes.empty:
        raise ValueError(
            "Svaki pair_id mora imati tačno "
            "dva patch-a."
        )

    pair_class_counts = (
        metadata.groupby("pair_id")[
            "target_class"
        ]
        .nunique()
    )

    if not (
        pair_class_counts == 2
    ).all():
        raise ValueError(
            "Svaki par mora sadržati defect "
            "i no_defect patch."
        )


def select_pairs(
    metadata,
    split_name,
    pairs_per_class,
    seed,
):
    split_metadata = metadata[
        metadata["split"] == split_name
    ].copy()

    if split_metadata.empty:
        raise ValueError(
            f"Nema patch-eva za split: {split_name}"
        )

    pair_table = (
        split_metadata[
            [
                "pair_id",
                "source_defect_class",
            ]
        ]
        .drop_duplicates("pair_id")
    )

    selected_parts = []

    for class_index, (
        defect_class,
        class_pairs,
    ) in enumerate(
        pair_table.groupby(
            "source_defect_class",
            sort=True,
        )
    ):
        sample_size = min(
            pairs_per_class,
            len(class_pairs),
        )

        sampled = class_pairs.sample(
            n=sample_size,
            random_state=seed + class_index,
        )

        selected_parts.append(sampled)

        print(
            f"{defect_class}: "
            f"{sample_size}/{len(class_pairs)} "
            "izabranih parova"
        )

    if not selected_parts:
        raise RuntimeError(
            "Nijedan par nije izabran."
        )

    selected_pairs = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    return selected_pairs.sort_values(
        [
            "source_defect_class",
            "pair_id",
        ]
    )


def crop_patch(row):
    image_path = Path(row.image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Slika ne postoji: {image_path}"
        )

    with Image.open(image_path) as image:
        image = image.convert("RGB")

        patch = image.crop(
            (
                int(row.xmin),
                int(row.ymin),
                int(row.xmax),
                int(row.ymax),
            )
        )

    return patch


def add_patch_to_axis(
    axis,
    row,
    column_title,
):
    patch = crop_patch(row)

    axis.imshow(patch)
    axis.set_xticks([])
    axis.set_yticks([])

    axis.set_title(
        (
            f"{column_title}\n"
            f"{row.patch_id}\n"
            f"{patch.width} × {patch.height}"
        ),
        fontsize=8,
    )

    border_color = (
        "crimson"
        if row.target_class == "defect"
        else "seagreen"
    )

    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(border_color)
        spine.set_linewidth(3)


def make_figure(
    metadata,
    selected_pairs,
    split_name,
    seed,
):
    number_of_pairs = len(selected_pairs)

    figure_height = max(
        4,
        number_of_pairs * 3.2,
    )

    figure, axes = plt.subplots(
        nrows=number_of_pairs,
        ncols=2,
        figsize=(12, figure_height),
        squeeze=False,
    )

    for row_index, selected_pair in enumerate(
        selected_pairs.itertuples(index=False)
    ):
        pair_rows = metadata[
            metadata["pair_id"]
            == selected_pair.pair_id
        ]

        defect_row = pair_rows[
            pair_rows["target_class"]
            == "defect"
        ].iloc[0]

        no_defect_row = pair_rows[
            pair_rows["target_class"]
            == "no_defect"
        ].iloc[0]

        add_patch_to_axis(
            axes[row_index, 0],
            defect_row,
            "DEFECT",
        )

        add_patch_to_axis(
            axes[row_index, 1],
            no_defect_row,
            "NO DEFECT kandidat",
        )

        axes[row_index, 0].set_ylabel(
            (
                f"Izvorna klasa:\n"
                f"{selected_pair.source_defect_class}\n"
                f"{selected_pair.pair_id}"
            ),
            fontsize=9,
            rotation=0,
            labelpad=75,
            verticalalignment="center",
        )

    figure.suptitle(
        (
            "Eksperiment 1 – vizuelna provera "
            f"patch parova\n"
            f"Split: {split_name}, seed: {seed}"
        ),
        fontsize=15,
        y=1.0,
    )

    figure.tight_layout()

    return figure


def main():
    arguments = parse_arguments()

    if arguments.pairs_per_class < 1:
        raise ValueError(
            "--pairs-per-class mora biti "
            "najmanje 1."
        )

    print("Čitanje patch metadata tabele...")

    metadata = pd.read_csv(
        METADATA_PATH
    )

    validate_metadata(metadata)

    selected_pairs = select_pairs(
        metadata=metadata,
        split_name=arguments.split,
        pairs_per_class=(
            arguments.pairs_per_class
        ),
        seed=arguments.seed,
    )

    figure = make_figure(
        metadata=metadata,
        selected_pairs=selected_pairs,
        split_name=arguments.split,
        seed=arguments.seed,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIRECTORY / (
        f"patch_pair_audit_{arguments.split}.jpg"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)

    print()
    print("Vizuelna provera je sačuvana:")
    print(output_path)
    print(
        "Broj prikazanih parova:",
        len(selected_pairs),
    )
    print(
        "Broj prikazanih patch-eva:",
        2 * len(selected_pairs),
    )


if __name__ == "__main__":
    main()