import argparse
import json
import time
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


CLASS_NAMES = [
    "abscission",
    "bulge",
    "corrosion",
    "crack",
    "leakage",
]

DATASET_ROOT = Path(
    "data/yolo_experiment_3"
)
OUTPUT_DIRECTORY = Path(
    "data/results/experiment_3_cascade"
)
TRAINING_DIRECTORY = (
    OUTPUT_DIRECTORY / "training"
)
TEST_DIRECTORY = (
    OUTPUT_DIRECTORY / "test_evaluation"
)
SUMMARY_JSON_PATH = (
    OUTPUT_DIRECTORY / "detector_results.json"
)
SUMMARY_CSV_PATH = (
    OUTPUT_DIRECTORY / "detector_results.csv"
)

BASE_MODEL = "yolo11n.pt"
RANDOM_SEED = 42


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Treniranje class-specific YOLO11n "
            "detektora za Eksperiment 3"
        )
    )

    parser.add_argument(
        "--class-name",
        choices=CLASS_NAMES,
        required=True,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--device",
        default="0",
        help=(
            "CUDA uređaj, na primer 0, "
            "ili cpu"
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
    )

    return parser.parse_args()


def extract_detection_metrics(metrics):
    box_metrics = metrics.box

    result = {
        "precision": float(
            box_metrics.mp
        ),
        "recall": float(
            box_metrics.mr
        ),
        "map50": float(
            box_metrics.map50
        ),
        "map75": float(
            box_metrics.map75
        ),
        "map50_95": float(
            box_metrics.map
        ),
    }

    if len(box_metrics.maps) > 0:
        result["class_map50_95"] = float(
            box_metrics.maps[0]
        )

    if hasattr(metrics, "speed"):
        result["speed_milliseconds"] = {
            str(key): float(value)
            for key, value in metrics.speed.items()
        }

    return result


def load_existing_results():
    if not SUMMARY_JSON_PATH.is_file():
        return {}

    with SUMMARY_JSON_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved = json.load(file)

    return saved.get(
        "detectors",
        {},
    )


def save_results(detectors):
    ordered_detectors = {
        class_name: detectors[class_name]
        for class_name in CLASS_NAMES
        if class_name in detectors
    }

    report = {
        "experiment": (
            "experiment_3_cascade_detection"
        ),
        "architecture": (
            "experiment_2_classifier_plus_"
            "class_specific_yolo11n"
        ),
        "base_model": BASE_MODEL,
        "random_seed": RANDOM_SEED,
        "selection_data": "validation",
        "final_evaluation_data": "test",
        "detectors": ordered_detectors,
    }

    with SUMMARY_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    rows = []

    for class_name, result in (
        ordered_detectors.items()
    ):
        test_metrics = result[
            "test_metrics"
        ]

        rows.append(
            {
                "class_name": class_name,
                "precision": test_metrics[
                    "precision"
                ],
                "recall": test_metrics[
                    "recall"
                ],
                "map50": test_metrics[
                    "map50"
                ],
                "map75": test_metrics[
                    "map75"
                ],
                "map50_95": test_metrics[
                    "map50_95"
                ],
                "epochs_requested": result[
                    "training_parameters"
                ]["epochs"],
                "training_time_seconds": (
                    result[
                        "training_time_seconds"
                    ]
                ),
                "best_model_path": result[
                    "best_model_path"
                ],
            }
        )

    pd.DataFrame(rows).to_csv(
        SUMMARY_CSV_PATH,
        index=False,
    )


def main():
    args = parse_arguments()

    class_name = args.class_name

    dataset_yaml = (
        DATASET_ROOT
        / class_name
        / "dataset.yaml"
    )

    if not dataset_yaml.is_file():
        raise FileNotFoundError(
            f"Nedostaje dataset YAML: {dataset_yaml}"
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    TRAINING_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    TEST_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_run_directory = (
        TRAINING_DIRECTORY / class_name
    )

    print(
        "Eksperiment 3 – class-specific "
        "YOLO detektor"
    )
    print("Klasa:", class_name)
    print("Osnovni model:", BASE_MODEL)
    print("Dataset:", dataset_yaml)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Image size:", args.image_size)
    print("Device:", args.device)

    model = YOLO(BASE_MODEL)

    training_start = time.perf_counter()

    model.train(
        data=str(
            dataset_yaml.resolve()
        ),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        project=str(
            TRAINING_DIRECTORY.resolve()
        ),
        name=class_name,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        seed=RANDOM_SEED,
        deterministic=True,
        amp=True,
        cache=False,
        plots=True,
        save=True,
        verbose=True,

        # Eksplicitno zabeležena YOLO
        # augmentacija tokom treninga.
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        cutmix=0.0,
        close_mosaic=10,
    )

    training_time = (
        time.perf_counter()
        - training_start
    )

    best_model_path = (
        training_run_directory
        / "weights"
        / "best.pt"
    )

    if not best_model_path.is_file():
        raise FileNotFoundError(
            "Trening nije napravio best.pt: "
            f"{best_model_path}"
        )

    print("\nNajbolji model:")
    print(best_model_path)

    # Test se poziva samo nakon što je
    # najbolji checkpoint izabran na validation skupu.
    best_model = YOLO(
        str(best_model_path)
    )

    test_start = time.perf_counter()

    test_metrics_object = best_model.val(
        data=str(
            dataset_yaml.resolve()
        ),
        split="test",
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        project=str(
            TEST_DIRECTORY.resolve()
        ),
        name=class_name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    test_time = (
        time.perf_counter()
        - test_start
    )

    test_metrics = (
        extract_detection_metrics(
            test_metrics_object
        )
    )

    detectors = load_existing_results()

    detectors[class_name] = {
        "class_name": class_name,
        "dataset_yaml": str(
            dataset_yaml
        ),
        "base_model": BASE_MODEL,
        "best_model_path": str(
            best_model_path
        ),
        "training_parameters": {
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "batch_size": int(
                args.batch_size
            ),
            "image_size": int(
                args.image_size
            ),
            "device": str(args.device),
            "optimizer": "auto",
            "seed": RANDOM_SEED,
        },
        "training_augmentation": {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.0,
            "cutmix": 0.0,
            "close_mosaic": 10,
        },
        "training_time_seconds": float(
            training_time
        ),
        "test_evaluation_time_seconds": float(
            test_time
        ),
        "test_metrics": test_metrics,
    }

    save_results(detectors)

    print("\nFinalni test rezultat:")
    print(
        f"Precision: "
        f"{test_metrics['precision']:.4f}"
    )
    print(
        f"Recall: "
        f"{test_metrics['recall']:.4f}"
    )
    print(
        f"mAP@0.50: "
        f"{test_metrics['map50']:.4f}"
    )
    print(
        f"mAP@0.75: "
        f"{test_metrics['map75']:.4f}"
    )
    print(
        f"mAP@0.50:0.95: "
        f"{test_metrics['map50_95']:.4f}"
    )

    print("\nSačuvano:")
    print(best_model_path)
    print(SUMMARY_JSON_PATH)
    print(SUMMARY_CSV_PATH)


if __name__ == "__main__":
    main()