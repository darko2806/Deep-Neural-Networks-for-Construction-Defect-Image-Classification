import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


CLASS_NAMES = [
    "abscission",
    "bulge",
    "corrosion",
    "crack",
    "leakage",
]

CLASS_TO_ORIGINAL_ID = {
    "crack": 0,
    "leakage": 1,
    "abscission": 2,
    "corrosion": 3,
    "bulge": 4,
}

METADATA_PATH = Path("data/metadata.csv")

CLASSIFIER_PREDICTIONS_PATH = Path(
    "data/results/experiment_2_multiclass/"
    "test_predictions.csv"
)

MODEL_DIRECTORY = Path(
    "data/results/experiment_3_cascade/training"
)

OUTPUT_DIRECTORY = Path(
    "data/results/experiment_3_cascade"
)

PREDICTIONS_PATH = (
    OUTPUT_DIRECTORY
    / "cascade_test_predictions.csv"
)

CLASS_METRICS_PATH = (
    OUTPUT_DIRECTORY
    / "cascade_class_metrics.csv"
)

RESULT_PATH = (
    OUTPUT_DIRECTORY
    / "cascade_test_results.json"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end evaluacija klasifikatora "
            "i class-specific YOLO detektora"
        )
    )

    parser.add_argument(
        "--device",
        default="0",
        help="CUDA uređaj, na primer 0, ili cpu",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
    )

    return parser.parse_args()


def xywh_to_xyxy(
    x_center,
    y_center,
    width,
    height,
):
    return [
        x_center - width / 2.0,
        y_center - height / 2.0,
        x_center + width / 2.0,
        y_center + height / 2.0,
    ]


def load_ground_truth_boxes(
    image_id,
    true_class,
):
    label_path = Path(
        f"MBDD2025/Labels/{image_id}.txt"
    )

    if not label_path.is_file():
        raise FileNotFoundError(
            f"Nedostaje labela: {label_path}"
        )

    expected_class_id = (
        CLASS_TO_ORIGINAL_ID[true_class]
    )

    boxes = []

    lines = label_path.read_text(
        encoding="utf-8"
    ).splitlines()

    for line in lines:
        if not line.strip():
            continue

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"Neispravna YOLO labela: "
                f"{label_path}"
            )

        class_id = int(parts[0])

        if class_id != expected_class_id:
            raise ValueError(
                f"Neočekivana klasa {class_id} "
                f"u fajlu {label_path}; "
                f"očekivano {expected_class_id}"
            )

        coordinates = list(
            map(float, parts[1:])
        )

        boxes.append(
            xywh_to_xyxy(*coordinates)
        )

    if not boxes:
        raise ValueError(
            f"Nema ground-truth box-eva: "
            f"{label_path}"
        )

    return np.asarray(
        boxes,
        dtype=np.float32,
    )


def calculate_iou(box_a, box_b):
    intersection_xmin = max(
        box_a[0],
        box_b[0],
    )

    intersection_ymin = max(
        box_a[1],
        box_b[1],
    )

    intersection_xmax = min(
        box_a[2],
        box_b[2],
    )

    intersection_ymax = min(
        box_a[3],
        box_b[3],
    )

    intersection_width = max(
        0.0,
        intersection_xmax
        - intersection_xmin,
    )

    intersection_height = max(
        0.0,
        intersection_ymax
        - intersection_ymin,
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area_a = (
        max(0.0, box_a[2] - box_a[0])
        * max(0.0, box_a[3] - box_a[1])
    )

    area_b = (
        max(0.0, box_b[2] - box_b[0])
        * max(0.0, box_b[3] - box_b[1])
    )

    union_area = (
        area_a
        + area_b
        - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return float(
        intersection_area / union_area
    )


def match_boxes(
    ground_truth_boxes,
    predicted_boxes,
    iou_threshold,
):
    candidates = []

    for ground_truth_index, ground_truth in enumerate(
        ground_truth_boxes
    ):
        for prediction_index, prediction in enumerate(
            predicted_boxes
        ):
            iou = calculate_iou(
                ground_truth,
                prediction,
            )

            if iou >= iou_threshold:
                candidates.append(
                    (
                        iou,
                        ground_truth_index,
                        prediction_index,
                    )
                )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    matched_ground_truth = set()
    matched_predictions = set()
    matches = []

    for (
        iou,
        ground_truth_index,
        prediction_index,
    ) in candidates:
        if (
            ground_truth_index
            in matched_ground_truth
        ):
            continue

        if (
            prediction_index
            in matched_predictions
        ):
            continue

        matched_ground_truth.add(
            ground_truth_index
        )

        matched_predictions.add(
            prediction_index
        )

        matches.append(
            {
                "ground_truth_index": int(
                    ground_truth_index
                ),
                "prediction_index": int(
                    prediction_index
                ),
                "iou": float(iou),
            }
        )

    true_positives = len(matches)

    false_positives = (
        len(predicted_boxes)
        - true_positives
    )

    false_negatives = (
        len(ground_truth_boxes)
        - true_positives
    )

    return {
        "true_positives": int(
            true_positives
        ),
        "false_positives": int(
            false_positives
        ),
        "false_negatives": int(
            false_negatives
        ),
        "matches": matches,
    }


def safe_divide(
    numerator,
    denominator,
):
    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


def clear_cuda_memory(
    model=None,
    prediction_stream=None,
):
    if prediction_stream is not None:
        close_method = getattr(
            prediction_stream,
            "close",
            None,
        )

        if callable(close_method):
            close_method()

    if model is not None:
        if getattr(
            model,
            "predictor",
            None,
        ) is not None:
            model.predictor = None

    del prediction_stream
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_evaluation_data():
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Nedostaje: {METADATA_PATH}"
        )

    if not CLASSIFIER_PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            "Nedostaju test predikcije "
            "Eksperimenta 2: "
            f"{CLASSIFIER_PREDICTIONS_PATH}"
        )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    classifier_predictions = pd.read_csv(
        CLASSIFIER_PREDICTIONS_PATH
    )

    required_metadata_columns = {
        "image_id",
        "image_path",
        "target_class",
        "visual_group",
        "split",
    }

    required_prediction_columns = {
        "image_id",
        "true_class",
        "predicted_class",
        "confidence",
    }

    missing_metadata_columns = (
        required_metadata_columns
        - set(metadata.columns)
    )

    missing_prediction_columns = (
        required_prediction_columns
        - set(classifier_predictions.columns)
    )

    if missing_metadata_columns:
        raise ValueError(
            "Metadata nema kolone: "
            f"{sorted(missing_metadata_columns)}"
        )

    if missing_prediction_columns:
        raise ValueError(
            "Predikcije nemaju kolone: "
            f"{sorted(missing_prediction_columns)}"
        )

    test_metadata = metadata.loc[
        metadata["split"].eq("test"),
        [
            "image_id",
            "image_path",
            "target_class",
            "visual_group",
        ],
    ].copy()

    test_metadata["image_id"] = (
        test_metadata["image_id"]
        .astype(str)
    )

    classifier_predictions["image_id"] = (
        classifier_predictions["image_id"]
        .astype(str)
    )

    evaluation = test_metadata.merge(
        classifier_predictions[
            [
                "image_id",
                "true_class",
                "predicted_class",
                "confidence",
            ]
        ],
        on="image_id",
        how="inner",
        validate="one_to_one",
    )

    if len(evaluation) != len(test_metadata):
        raise ValueError(
            "Nedostaju predikcije klasifikatora "
            "za neke test slike"
        )

    if len(evaluation) != len(
        classifier_predictions
    ):
        raise ValueError(
            "Broj predikcija klasifikatora "
            "nije jednak broju test slika"
        )

    if not (
        evaluation["target_class"]
        == evaluation["true_class"]
    ).all():
        raise ValueError(
            "Klase iz metadata fajla i "
            "predikcija klasifikatora "
            "nisu jednake"
        )

    invalid_true_classes = (
        set(evaluation["true_class"])
        - set(CLASS_NAMES)
    )

    invalid_predicted_classes = (
        set(evaluation["predicted_class"])
        - set(CLASS_NAMES)
    )

    if invalid_true_classes:
        raise ValueError(
            "Nepoznate stvarne klase: "
            f"{sorted(invalid_true_classes)}"
        )

    if invalid_predicted_classes:
        raise ValueError(
            "Nepoznate predviđene klase: "
            f"{sorted(invalid_predicted_classes)}"
        )

    missing_images = [
        path
        for path in evaluation["image_path"]
        if not Path(path).is_file()
    ]

    if missing_images:
        raise FileNotFoundError(
            "Nedostaju slike. Prvi primer: "
            f"{missing_images[0]}"
        )

    return evaluation


def verify_detector_models():
    for class_name in CLASS_NAMES:
        model_path = (
            MODEL_DIRECTORY
            / class_name
            / "weights"
            / "best.pt"
        )

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Nedostaje model: {model_path}"
            )


def evaluate_detectors(
    evaluation,
    args,
):
    result_records = []
    start_time = time.perf_counter()

    for detector_class in CLASS_NAMES:
        class_rows = evaluation.loc[
            evaluation[
                "predicted_class"
            ].eq(detector_class)
        ].copy()

        class_rows = class_rows.reset_index(
            drop=True
        )

        print(
            f"\nDetektor: {detector_class}"
        )

        print(
            "Broj rutiranih slika:",
            len(class_rows),
        )

        if class_rows.empty:
            continue

        model_path = (
            MODEL_DIRECTORY
            / detector_class
            / "weights"
            / "best.pt"
        )

        model = None
        processed_predictions = 0

        try:
            model = YOLO(
                str(model_path)
            )

            number_of_images = len(
                class_rows
            )

            number_of_batches = (
                number_of_images
                + args.batch_size
                - 1
            ) // args.batch_size

            for batch_index, batch_start in enumerate(
                range(
                    0,
                    number_of_images,
                    args.batch_size,
                ),
                start=1,
            ):
                batch_end = min(
                    batch_start
                    + args.batch_size,
                    number_of_images,
                )

                batch_rows = (
                    class_rows.iloc[
                        batch_start:batch_end
                    ]
                )

                batch_paths = (
                    batch_rows["image_path"]
                    .astype(str)
                    .tolist()
                )

                prediction_stream = model.predict(
                    source=batch_paths,
                    conf=args.confidence,
                    iou=0.70,
                    imgsz=args.image_size,
                    device=args.device,
                    save=False,
                    verbose=False,
                    stream=True,
                )

                processed_in_batch = 0

                try:
                    for row, prediction in zip(
                        batch_rows.itertuples(
                            index=False
                        ),
                        prediction_stream,
                    ):
                        processed_in_batch += 1
                        processed_predictions += 1

                        true_class = str(
                            row.true_class
                        )

                        predicted_class = str(
                            row.predicted_class
                        )

                        ground_truth_boxes = (
                            load_ground_truth_boxes(
                                image_id=row.image_id,
                                true_class=true_class,
                            )
                        )

                        if (
                            prediction.boxes is None
                            or len(
                                prediction.boxes
                            ) == 0
                        ):
                            predicted_boxes = (
                                np.empty(
                                    (0, 4),
                                    dtype=np.float32,
                                )
                            )

                            detection_confidences = (
                                np.empty(
                                    0,
                                    dtype=np.float32,
                                )
                            )
                        else:
                            predicted_boxes = (
                                prediction.boxes.xyxy
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.float32)
                            )

                            original_height = float(
                                prediction.orig_shape[0]
                            )

                            original_width = float(
                                prediction.orig_shape[1]
                            )

                            predicted_boxes[
                                :, [0, 2]
                            ] /= original_width

                            predicted_boxes[
                                :, [1, 3]
                            ] /= original_height

                            predicted_boxes = (
                                np.clip(
                                    predicted_boxes,
                                    0.0,
                                    1.0,
                                )
                            )

                            detection_confidences = (
                                prediction.boxes.conf
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.float32)
                            )

                        classification_correct = (
                            true_class
                            == predicted_class
                        )

                        if classification_correct:
                            matching = match_boxes(
                                ground_truth_boxes,
                                predicted_boxes,
                                args.iou_threshold,
                            )
                        else:
                            matching = {
                                "true_positives": 0,
                                "false_positives": int(
                                    len(
                                        predicted_boxes
                                    )
                                ),
                                "false_negatives": int(
                                    len(
                                        ground_truth_boxes
                                    )
                                ),
                                "matches": [],
                            }

                        true_positives = matching[
                            "true_positives"
                        ]

                        false_positives = matching[
                            "false_positives"
                        ]

                        false_negatives = matching[
                            "false_negatives"
                        ]

                        localization_success = bool(
                            classification_correct
                            and true_positives > 0
                        )

                        all_objects_detected = bool(
                            classification_correct
                            and false_negatives == 0
                            and len(
                                ground_truth_boxes
                            ) > 0
                        )

                        result_records.append(
                            {
                                "image_id": str(
                                    row.image_id
                                ),
                                "image_path": str(
                                    row.image_path
                                ),
                                "visual_group": str(
                                    row.visual_group
                                ),
                                "true_class": (
                                    true_class
                                ),
                                "predicted_class": (
                                    predicted_class
                                ),
                                "classifier_confidence": (
                                    float(
                                        row.confidence
                                    )
                                ),
                                "classification_correct": (
                                    classification_correct
                                ),
                                "ground_truth_box_count": (
                                    int(
                                        len(
                                            ground_truth_boxes
                                        )
                                    )
                                ),
                                "predicted_box_count": (
                                    int(
                                        len(
                                            predicted_boxes
                                        )
                                    )
                                ),
                                "true_positives": int(
                                    true_positives
                                ),
                                "false_positives": int(
                                    false_positives
                                ),
                                "false_negatives": int(
                                    false_negatives
                                ),
                                "localization_success": (
                                    localization_success
                                ),
                                "all_objects_detected": (
                                    all_objects_detected
                                ),
                                "ground_truth_boxes": (
                                    json.dumps(
                                        ground_truth_boxes
                                        .tolist()
                                    )
                                ),
                                "predicted_boxes": (
                                    json.dumps(
                                        predicted_boxes
                                        .tolist()
                                    )
                                ),
                                "detection_confidences": (
                                    json.dumps(
                                        detection_confidences
                                        .tolist()
                                    )
                                ),
                                "matches": (
                                    json.dumps(
                                        matching[
                                            "matches"
                                        ]
                                    )
                                ),
                            }
                        )

                        del prediction
                        del ground_truth_boxes
                        del predicted_boxes
                        del detection_confidences
                        del matching

                finally:
                    close_method = getattr(
                        prediction_stream,
                        "close",
                        None,
                    )

                    if callable(close_method):
                        close_method()

                    del prediction_stream

                    if getattr(
                        model,
                        "predictor",
                        None,
                    ) is not None:
                        model.predictor.results = None
                        model.predictor.batch = None
                        model.predictor.dataset = None

                    gc.collect()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if (
                    processed_in_batch
                    != len(batch_rows)
                ):
                    raise ValueError(
                        f"Detektor "
                        f"{detector_class}, "
                        f"batch {batch_index}: "
                        f"obrađeno "
                        f"{processed_in_batch}, "
                        f"očekivano "
                        f"{len(batch_rows)}"
                    )

                if (
                    batch_index % 25 == 0
                    or batch_index
                    == number_of_batches
                ):
                    print(
                        "Obrađen batch "
                        f"{batch_index}/"
                        f"{number_of_batches}"
                    )

            if (
                processed_predictions
                != number_of_images
            ):
                raise ValueError(
                    f"Detektor {detector_class}: "
                    f"obrađeno "
                    f"{processed_predictions}, "
                    f"očekivano "
                    f"{number_of_images} slika"
                )

        finally:
            if model is not None:
                if getattr(
                    model,
                    "predictor",
                    None,
                ) is not None:
                    model.predictor = None

                del model

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(
            "Ukupno obrađeno:",
            processed_predictions,
        )

    inference_time = (
        time.perf_counter()
        - start_time
    )

    results = pd.DataFrame(
        result_records
    )

    results = results.sort_values(
        "image_id"
    ).reset_index(drop=True)

    if len(results) != len(evaluation):
        raise ValueError(
            "Nisu evaluirane sve test slike. "
            f"Evaluirano {len(results)}, "
            f"očekivano {len(evaluation)}."
        )

    return results, inference_time


def calculate_class_metrics(results):
    statistics = {
        class_name: {
            "support_images": 0,
            "routed_images": 0,
            "ground_truth_boxes": 0,
            "predicted_boxes": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }
        for class_name in CLASS_NAMES
    }

    for row in results.itertuples(
        index=False
    ):
        true_class = str(
            row.true_class
        )

        predicted_class = str(
            row.predicted_class
        )

        statistics[
            true_class
        ]["support_images"] += 1

        statistics[
            predicted_class
        ]["routed_images"] += 1

        statistics[
            true_class
        ]["ground_truth_boxes"] += int(
            row.ground_truth_box_count
        )

        statistics[
            predicted_class
        ]["predicted_boxes"] += int(
            row.predicted_box_count
        )

        if true_class == predicted_class:
            statistics[
                true_class
            ]["true_positives"] += int(
                row.true_positives
            )

            statistics[
                true_class
            ]["false_positives"] += int(
                row.false_positives
            )

            statistics[
                true_class
            ]["false_negatives"] += int(
                row.false_negatives
            )
        else:
            statistics[
                predicted_class
            ]["false_positives"] += int(
                row.predicted_box_count
            )

            statistics[
                true_class
            ]["false_negatives"] += int(
                row.ground_truth_box_count
            )

    class_rows = []

    for class_name in CLASS_NAMES:
        values = statistics[
            class_name
        ]

        precision = safe_divide(
            values["true_positives"],
            (
                values["true_positives"]
                + values["false_positives"]
            ),
        )

        recall = safe_divide(
            values["true_positives"],
            (
                values["true_positives"]
                + values["false_negatives"]
            ),
        )

        f1 = safe_divide(
            2.0 * precision * recall,
            precision + recall,
        )

        class_rows.append(
            {
                "class_name": class_name,
                **values,
                "cascade_precision_iou50": (
                    precision
                ),
                "cascade_recall_iou50": recall,
                "cascade_f1_iou50": f1,
            }
        )

    return pd.DataFrame(
        class_rows
    )


def create_summary(
    results,
    class_metrics,
    inference_time,
    args,
):
    total_true_positives = int(
        class_metrics[
            "true_positives"
        ].sum()
    )

    total_false_positives = int(
        class_metrics[
            "false_positives"
        ].sum()
    )

    total_false_negatives = int(
        class_metrics[
            "false_negatives"
        ].sum()
    )

    object_precision = safe_divide(
        total_true_positives,
        (
            total_true_positives
            + total_false_positives
        ),
    )

    object_recall = safe_divide(
        total_true_positives,
        (
            total_true_positives
            + total_false_negatives
        ),
    )

    object_f1 = safe_divide(
        2.0
        * object_precision
        * object_recall,
        object_precision
        + object_recall,
    )

    correctly_classified = results[
        "classification_correct"
    ].astype(bool)

    if correctly_classified.any():
        conditional_localization_success = (
            results.loc[
                correctly_classified,
                "localization_success",
            ]
            .astype(bool)
            .mean()
        )
    else:
        conditional_localization_success = 0.0

    summary = {
        "experiment": (
            "experiment_3_end_to_end_cascade"
        ),
        "pipeline": (
            "inception_resnet_v2_xgboost_"
            "classifier_plus_"
            "class_specific_yolo11n"
        ),
        "test_set_previously_used": True,
        "confidence_threshold": float(
            args.confidence
        ),
        "matching_iou_threshold": float(
            args.iou_threshold
        ),
        "image_size": int(
            args.image_size
        ),
        "batch_size": int(
            args.batch_size
        ),
        "device": str(
            args.device
        ),
        "sample_counts": {
            "test_images": int(
                len(results)
            ),
            "correctly_classified_images": (
                int(
                    correctly_classified.sum()
                )
            ),
            "ground_truth_boxes": int(
                results[
                    "ground_truth_box_count"
                ].sum()
            ),
            "predicted_boxes": int(
                results[
                    "predicted_box_count"
                ].sum()
            ),
        },
        "classification_accuracy": float(
            correctly_classified.mean()
        ),
        "conditional_localization_success": (
            float(
                conditional_localization_success
            )
        ),
        "end_to_end_image_success": float(
            results[
                "localization_success"
            ]
            .astype(bool)
            .mean()
        ),
        "all_objects_detected_image_rate": (
            float(
                results[
                    "all_objects_detected"
                ]
                .astype(bool)
                .mean()
            )
        ),
        "class_aware_object_metrics_iou50": {
            "true_positives": (
                total_true_positives
            ),
            "false_positives": (
                total_false_positives
            ),
            "false_negatives": (
                total_false_negatives
            ),
            "precision": float(
                object_precision
            ),
            "recall": float(
                object_recall
            ),
            "f1": float(
                object_f1
            ),
        },
        "inference_time_seconds": float(
            inference_time
        ),
        "predictions_path": str(
            PREDICTIONS_PATH
        ),
        "class_metrics_path": str(
            CLASS_METRICS_PATH
        ),
    }

    return summary


def save_results(
    results,
    class_metrics,
    summary,
):
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    class_metrics.to_csv(
        CLASS_METRICS_PATH,
        index=False,
    )

    with RESULT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )


def print_results(
    results,
    class_metrics,
    summary,
):
    object_metrics = summary[
        "class_aware_object_metrics_iou50"
    ]

    print(
        "\nEksperiment 3 – rezultat cele kaskade"
    )

    print(
        "Test slika:",
        len(results),
    )

    print(
        "Classification accuracy:",
        f"{summary['classification_accuracy']:.4f}",
    )

    print(
        "Lokalizacija kada je klasa tačna:",
        (
            f"{summary[
                'conditional_localization_success'
            ]:.4f}"
        ),
    )

    print(
        "End-to-end uspeh po slici:",
        (
            f"{summary[
                'end_to_end_image_success'
            ]:.4f}"
        ),
    )

    print(
        "Svi objekti pronađeni:",
        (
            f"{summary[
                'all_objects_detected_image_rate'
            ]:.4f}"
        ),
    )

    print(
        "\nClass-aware object metrike "
        "@ IoU 0.50:"
    )

    print(
        "Precision:",
        f"{object_metrics['precision']:.4f}",
    )

    print(
        "Recall:",
        f"{object_metrics['recall']:.4f}",
    )

    print(
        "F1:",
        f"{object_metrics['f1']:.4f}",
    )

    print(
        "\nRezultat po klasama:"
    )

    print(
        class_metrics.to_string(
            index=False
        )
    )

    print(
        "\nUkupno vreme inferencije:",
        (
            f"{summary[
                'inference_time_seconds'
            ]:.1f} s"
        ),
    )

    print(
        "\nSačuvano:"
    )

    print(RESULT_PATH)
    print(PREDICTIONS_PATH)
    print(CLASS_METRICS_PATH)


def main():
    args = parse_arguments()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Eksperiment 3 – end-to-end "
        "cascade evaluacija"
    )

    print(
        "Uređaj:",
        args.device,
    )

    print(
        "Batch size:",
        args.batch_size,
    )

    print(
        "Confidence threshold:",
        args.confidence,
    )

    print(
        "IoU prag za poklapanje:",
        args.iou_threshold,
    )

    evaluation = load_evaluation_data()

    verify_detector_models()

    print(
        "Broj test slika:",
        len(evaluation),
    )

    print(
        "\nRaspodela predviđenih klasa:"
    )

    print(
        evaluation[
            "predicted_class"
        ].value_counts()
    )

    results, inference_time = (
        evaluate_detectors(
            evaluation=evaluation,
            args=args,
        )
    )

    class_metrics = (
        calculate_class_metrics(
            results
        )
    )

    summary = create_summary(
        results=results,
        class_metrics=class_metrics,
        inference_time=inference_time,
        args=args,
    )

    save_results(
        results=results,
        class_metrics=class_metrics,
        summary=summary,
    )

    print_results(
        results=results,
        class_metrics=class_metrics,
        summary=summary,
    )


if __name__ == "__main__":
    main()