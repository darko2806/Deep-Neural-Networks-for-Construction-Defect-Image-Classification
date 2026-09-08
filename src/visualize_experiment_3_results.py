import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image


RESULT_DIRECTORY = Path(
    "data/results/experiment_3_cascade"
)
PREDICTIONS_PATH = (
    RESULT_DIRECTORY / "cascade_test_predictions.csv"
)
DETECTOR_RESULTS_PATH = (
    RESULT_DIRECTORY / "detector_results.csv"
)
CASCADE_METRICS_PATH = (
    RESULT_DIRECTORY / "cascade_class_metrics.csv"
)

CLASS_ORDER = [
    "abscission",
    "bulge",
    "corrosion",
    "crack",
    "leakage",
]

CLASS_LABELS = {
    "abscission": "Abscission",
    "bulge": "Bulge",
    "corrosion": "Corrosion",
    "crack": "Crack",
    "leakage": "Leakage",
}

CLASS_COLORS = {
    "abscission": "#E45756",
    "bulge": "#F2CF5B",
    "corrosion": "#54A24B",
    "crack": "#4C78A8",
    "leakage": "#B279A2",
}

GROUND_TRUTH_COLOR = "#2CA02C"
PREDICTION_COLOR = "#FF7F0E"


def parse_json_list(value):
    if pd.isna(value) or value == "":
        return []
    return json.loads(value)


def normalize_boolean_columns(frame):
    for column in [
        "classification_correct",
        "localization_success",
        "all_objects_detected",
    ]:
        if frame[column].dtype != bool:
            frame[column] = (
                frame[column]
                .astype(str)
                .str.lower()
                .map({"true": True, "false": False})
            )
    return frame


def best_match_iou(matches):
    parsed = parse_json_list(matches)
    if not parsed:
        return 0.0
    return max(float(item["iou"]) for item in parsed)


def draw_boxes(ax, row):
    image = Image.open(row.image_path).convert("RGB")
    width, height = image.size
    ax.imshow(image)

    ground_truth_boxes = parse_json_list(
        row.ground_truth_boxes
    )
    predicted_boxes = parse_json_list(
        row.predicted_boxes
    )
    confidences = parse_json_list(
        row.detection_confidences
    )

    for index, box in enumerate(ground_truth_boxes):
        xmin, ymin, xmax, ymax = box
        rectangle = Rectangle(
            (xmin * width, ymin * height),
            (xmax - xmin) * width,
            (ymax - ymin) * height,
            fill=False,
            edgecolor=GROUND_TRUTH_COLOR,
            linewidth=2.0,
        )
        ax.add_patch(rectangle)
        if index == 0:
            ax.text(
                xmin * width,
                max(0, ymin * height - 4),
                "GT",
                color="white",
                fontsize=7,
                bbox={
                    "facecolor": GROUND_TRUTH_COLOR,
                    "alpha": 0.9,
                    "pad": 1.5,
                    "edgecolor": "none",
                },
            )

    for index, box in enumerate(predicted_boxes):
        xmin, ymin, xmax, ymax = box
        rectangle = Rectangle(
            (xmin * width, ymin * height),
            (xmax - xmin) * width,
            (ymax - ymin) * height,
            fill=False,
            edgecolor=PREDICTION_COLOR,
            linewidth=1.7,
            linestyle="--",
        )
        ax.add_patch(rectangle)
        confidence = (
            confidences[index]
            if index < len(confidences)
            else 0.0
        )
        if index == 0:
            ax.text(
                xmin * width,
                min(height - 10, ymax * height + 12),
                f"Pred {confidence:.2f}",
                color="white",
                fontsize=7,
                bbox={
                    "facecolor": PREDICTION_COLOR,
                    "alpha": 0.9,
                    "pad": 1.5,
                    "edgecolor": "none",
                },
            )

    ax.set_axis_off()


def create_detector_metrics_figure(detectors):
    detectors = (
        detectors.set_index("class_name")
        .reindex(CLASS_ORDER)
        .reset_index()
    )

    metrics = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("map50", "mAP@0.50"),
        ("map50_95", "mAP@0.50:0.95"),
    ]

    x = np.arange(len(CLASS_ORDER))
    width = 0.19

    figure, ax = plt.subplots(figsize=(13, 6.5))

    for index, (column, label) in enumerate(metrics):
        offset = (index - 1.5) * width
        ax.bar(
            x + offset,
            detectors[column],
            width=width,
            label=label,
        )

    ax.set_title(
        "Eksperiment 3 – samostalni rezultati YOLO detektora na test skupu",
        fontsize=14,
        pad=14,
    )
    ax.set_ylabel("Vrednost metrike")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [CLASS_LABELS[name] for name in CLASS_ORDER]
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, loc="upper center")
    figure.tight_layout()

    output_path = (
        RESULT_DIRECTORY / "detector_test_metrics.png"
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def create_cascade_metrics_figure(metrics):
    metrics = (
        metrics.set_index("class_name")
        .reindex(CLASS_ORDER)
        .reset_index()
    )

    columns = [
        ("cascade_precision_iou50", "Precision"),
        ("cascade_recall_iou50", "Recall"),
        ("cascade_f1_iou50", "F1"),
    ]

    x = np.arange(len(CLASS_ORDER))
    width = 0.25
    figure, ax = plt.subplots(figsize=(12, 6.5))

    for index, (column, label) in enumerate(columns):
        bars = ax.bar(
            x + (index - 1) * width,
            metrics[column],
            width=width,
            label=label,
        )
        ax.bar_label(
            bars,
            fmt="%.2f",
            fontsize=8,
            padding=2,
        )

    ax.set_title(
        "Eksperiment 3 – class-aware metrike cele kaskade pri IoU 0,50",
        fontsize=14,
        pad=14,
    )
    ax.set_ylabel("Vrednost metrike")
    ax.set_ylim(0.0, 1.10)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [CLASS_LABELS[name] for name in CLASS_ORDER]
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, loc="upper center")
    figure.tight_layout()

    output_path = (
        RESULT_DIRECTORY / "cascade_class_metrics.png"
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def select_success_examples(predictions):
    candidates = predictions.loc[
        predictions["classification_correct"]
        & predictions["localization_success"]
    ].copy()

    candidates["best_iou"] = candidates[
        "matches"
    ].map(best_match_iou)

    selected = []
    for class_name in CLASS_ORDER:
        class_candidates = candidates.loc[
            candidates["true_class"].eq(class_name)
        ].copy()

        class_candidates["box_distance"] = (
            class_candidates["ground_truth_box_count"] - 2
        ).abs()

        class_candidates = class_candidates.sort_values(
            ["best_iou", "box_distance", "classifier_confidence"],
            ascending=[False, True, False],
        )

        unique_candidates = class_candidates.drop_duplicates(
            "visual_group"
        )

        chosen = unique_candidates.head(2)

        if len(chosen) < 2:
            remaining = class_candidates.loc[
                ~class_candidates["image_id"].isin(
                    chosen["image_id"]
                )
            ].head(2 - len(chosen))
            chosen = pd.concat([chosen, remaining])

        selected.extend(chosen.to_dict("records"))

    return pd.DataFrame(selected)


def create_success_figure(predictions):
    selected = select_success_examples(predictions)
    figure, axes = plt.subplots(
        len(CLASS_ORDER),
        2,
        figsize=(14, 17),
        squeeze=False,
    )

    for row_index, class_name in enumerate(CLASS_ORDER):
        examples = selected.loc[
            selected["true_class"].eq(class_name)
        ]

        for column_index, row in enumerate(
            examples.itertuples(index=False)
        ):
            ax = axes[row_index, column_index]
            draw_boxes(ax, row)
            ax.set_title(
                f"{CLASS_LABELS[class_name]} – {row.image_id}\n"
                f"TP={row.true_positives}, FP={row.false_positives}, "
                f"FN={row.false_negatives}",
                fontsize=10,
            )

    figure.suptitle(
        "Uspešni primeri cele kaskade\n"
        "zelena puna linija: ground truth; narandžasta isprekidana: predikcija",
        fontsize=15,
        y=0.995,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.98])

    output_path = (
        RESULT_DIRECTORY / "cascade_success_examples.jpg"
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def create_failure_figure(predictions):
    classifier_errors = predictions.loc[
        ~predictions["classification_correct"]
    ].sort_values(
        "classifier_confidence",
        ascending=False,
    ).drop_duplicates(
        "visual_group"
    ).head(3)

    localization_misses = predictions.loc[
        predictions["classification_correct"]
        & ~predictions["localization_success"]
    ].sort_values(
        ["ground_truth_box_count", "classifier_confidence"],
        ascending=[False, False],
    ).drop_duplicates(
        "visual_group"
    ).head(3)

    selected = pd.concat(
        [classifier_errors, localization_misses],
        ignore_index=True,
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9),
        squeeze=False,
    )

    for index, row in enumerate(
        selected.itertuples(index=False)
    ):
        ax = axes.flat[index]
        draw_boxes(ax, row)

        if row.classification_correct:
            reason = "tačna klasa, lokalizacija promašena"
        else:
            reason = (
                f"pogrešna klasa: {row.true_class} → "
                f"{row.predicted_class}"
            )

        ax.set_title(
            f"{row.image_id}\n{reason}",
            fontsize=10,
        )

    figure.suptitle(
        "Reprezentativne greške kaskadnog sistema",
        fontsize=15,
        y=0.99,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = (
        RESULT_DIRECTORY / "cascade_failure_examples.jpg"
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main():
    for path in [
        PREDICTIONS_PATH,
        DETECTOR_RESULTS_PATH,
        CASCADE_METRICS_PATH,
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Nedostaje: {path}")

    predictions = normalize_boolean_columns(
        pd.read_csv(PREDICTIONS_PATH)
    )
    detectors = pd.read_csv(DETECTOR_RESULTS_PATH)
    cascade_metrics = pd.read_csv(CASCADE_METRICS_PATH)

    outputs = [
        create_detector_metrics_figure(detectors),
        create_cascade_metrics_figure(cascade_metrics),
        create_success_figure(predictions),
        create_failure_figure(predictions),
    ]

    print("Generisane vizualizacije:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
