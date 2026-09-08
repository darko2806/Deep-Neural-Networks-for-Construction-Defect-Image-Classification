import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


MANIFEST_PATH = Path(
    "data/results/experiment_3_cascade/"
    "detection_dataset_manifest.csv"
)
OUTPUT_DIRECTORY = Path(
    "data/results/experiment_3_cascade"
)

CLASS_ORDER = [
    "abscission",
    "bulge",
    "corrosion",
    "crack",
    "leakage",
]

CLASS_COLORS = {
    "abscission": "#E45756",
    "bulge": "#F2CF5B",
    "corrosion": "#54A24B",
    "crack": "#4C78A8",
    "leakage": "#B279A2",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Vizuelna provera YOLO anotacija "
            "za Eksperiment 3"
        )
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "validation",
            "test",
        ],
        default="validation",
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def read_yolo_boxes(
    label_path,
    image_width,
    image_height,
):
    boxes = []

    for line in Path(label_path).read_text(
        encoding="utf-8"
    ).splitlines():
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"Neispravna labela: {label_path}"
            )

        class_id = int(parts[0])

        if class_id != 0:
            raise ValueError(
                "Class-specific labela mora imati "
                f"lokalnu klasu 0: {label_path}"
            )

        (
            x_center,
            y_center,
            box_width,
            box_height,
        ) = map(float, parts[1:])

        xmin = (
            x_center - box_width / 2
        ) * image_width
        ymin = (
            y_center - box_height / 2
        ) * image_height
        xmax = (
            x_center + box_width / 2
        ) * image_width
        ymax = (
            y_center + box_height / 2
        ) * image_height

        boxes.append(
            (
                max(0.0, xmin),
                max(0.0, ymin),
                min(float(image_width), xmax),
                min(float(image_height), ymax),
            )
        )

    return boxes


def draw_boxes(
    image,
    boxes,
    class_name,
):
    annotated = image.copy()
    draw = ImageDraw.Draw(
        annotated
    )

    line_width = max(
        3,
        round(
            min(image.size) / 180
        ),
    )

    color = CLASS_COLORS[
        class_name
    ]

    for box_index, box in enumerate(
        boxes,
        start=1,
    ):
        draw.rectangle(
            box,
            outline=color,
            width=line_width,
        )

        label = (
            f"{class_name} {box_index}"
        )

        text_x = box[0]
        text_y = max(
            0,
            box[1] - 18,
        )

        text_box = draw.textbbox(
            (text_x, text_y),
            label,
        )

        draw.rectangle(
            text_box,
            fill=color,
        )
        draw.text(
            (text_x, text_y),
            label,
            fill="white",
        )

    return annotated


def main():
    args = parse_arguments()

    if args.images_per_class < 1:
        raise ValueError(
            "images-per-class mora biti najmanje 1"
        )

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Nedostaje: {MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    split_manifest = manifest.loc[
        manifest["split"].eq(
            args.split
        )
    ].copy()

    selected_rows = []

    for class_index, class_name in enumerate(
        CLASS_ORDER
    ):
        class_rows = split_manifest.loc[
            split_manifest[
                "detector_class"
            ].eq(class_name)
        ]

        if len(class_rows) < args.images_per_class:
            raise ValueError(
                f"Nema dovoljno slika za {class_name}"
            )

        selected = class_rows.sample(
            n=args.images_per_class,
            random_state=(
                args.seed + class_index
            ),
        )

        selected_rows.extend(
            selected.to_dict("records")
        )

    figure, axes = plt.subplots(
        nrows=len(CLASS_ORDER),
        ncols=args.images_per_class,
        figsize=(
            5 * args.images_per_class,
            4 * len(CLASS_ORDER),
        ),
        squeeze=False,
    )

    for row_index, class_name in enumerate(
        CLASS_ORDER
    ):
        class_examples = [
            row
            for row in selected_rows
            if row["detector_class"]
            == class_name
        ]

        for column_index, row in enumerate(
            class_examples
        ):
            image_path = Path(
                row["source_image_path"]
            )
            label_path = Path(
                row["yolo_label_path"]
            )

            with Image.open(
                image_path
            ) as image:
                image = image.convert(
                    "RGB"
                )

                boxes = read_yolo_boxes(
                    label_path,
                    image.width,
                    image.height,
                )

                annotated = draw_boxes(
                    image,
                    boxes,
                    class_name,
                )

            axis = axes[
                row_index,
                column_index,
            ]

            axis.imshow(annotated)
            axis.set_title(
                f"{class_name}\n"
                f"{row['image_id']} – "
                f"{len(boxes)} box-eva",
                fontsize=10,
            )
            axis.axis("off")

    figure.suptitle(
        "Eksperiment 3 – vizuelna provera "
        f"YOLO anotacija ({args.split})",
        fontsize=16,
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.98]
    )

    output_path = (
        OUTPUT_DIRECTORY
        / f"yolo_annotation_audit_{args.split}.jpg"
    )

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(
        "Vizuelna provera je sačuvana:"
    )
    print(output_path)
    print(
        "Broj prikazanih slika:",
        len(selected_rows),
    )


if __name__ == "__main__":
    main()