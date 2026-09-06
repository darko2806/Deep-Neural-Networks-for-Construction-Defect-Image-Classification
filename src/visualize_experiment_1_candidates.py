import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


AUDIT_PATH = Path(
    "data/results/experiment_1_binary/"
    "dataset_audit.json"
)

IMAGES_DIRECTORY = Path(
    "MBDD2025/JPEGImages"
)

OUTPUT_PATH = Path(
    "data/results/experiment_1_binary/"
    "negative_candidates.jpg"
)


def find_image_path(image_id):
    matches = sorted(
        IMAGES_DIRECTORY.glob(f"{image_id}.*")
    )

    if not matches:
        raise FileNotFoundError(
            f"Slika nije pronađena: {image_id}"
        )

    return matches[0]


def main():
    with AUDIT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        audit = json.load(file)

    candidate_ids = audit[
        "negative_candidates"
    ]["image_ids"]

    if not candidate_ids:
        raise ValueError(
            "Nema potencijalnih negativnih slika."
        )

    columns = 2
    rows = (
        len(candidate_ids) + columns - 1
    ) // columns

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, rows * 5),
    )

    axes = axes.flatten()

    for axis, image_id in zip(
        axes,
        candidate_ids,
    ):
        image_path = find_image_path(image_id)

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            width, height = rgb_image.size
            axis.imshow(rgb_image)

        axis.set_title(
            f"{image_id}\n"
            f"prazna anotacija, "
            f"{width} × {height}",
            fontsize=11,
        )
        axis.axis("off")

        print(
            image_id,
            image_path,
            f"{width}x{height}",
        )

    for axis in axes[len(candidate_ids):]:
        axis.axis("off")

    figure.suptitle(
        "Eksperiment 1 – potencijalne "
        "No Defect slike",
        fontsize=16,
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.97)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("\nSačuvano:", OUTPUT_PATH)


if __name__ == "__main__":
    main()