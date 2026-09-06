import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.feature_data import (
    load_feature_splits,
)
from src.feature_extraction import (
    FEATURE_EXTRACTOR_MODELS,
)


MAX_DEPTH_VALUES = [
    4,
    8,
]

LEARNING_RATE_VALUES = [
    0.03,
    0.1,
]

MIN_CHILD_WEIGHT_VALUES = [
    1,
    5,
]

MAX_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 50

FEATURE_DIRECTORY = Path(
    "data/features/experiment_1_binary"
)

OUTPUT_DIRECTORY = Path(
    "data/results/experiment_1_binary/"
    "xgboost"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "XGBoost za patch-level "
            "binarnu klasifikaciju."
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
        "--device",
        choices=[
            "cuda",
            "cpu",
        ],
        default="cuda",
    )

    return parser.parse_args()


def calculate_metrics(
    y_true,
    y_predicted,
):
    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_predicted,
        ),
        "balanced_accuracy": (
            balanced_accuracy_score(
                y_true,
                y_predicted,
            )
        ),
        "macro_f1": f1_score(
            y_true,
            y_predicted,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_predicted,
            average="weighted",
            zero_division=0,
        ),
        "defect_f1": f1_score(
            y_true,
            y_predicted,
            pos_label="defect",
            average="binary",
            zero_division=0,
        ),
    }

    return {
        metric_name: float(metric_value)
        for metric_name, metric_value
        in metrics.items()
    }


def create_parameter_combinations():
    combinations = product(
        MAX_DEPTH_VALUES,
        LEARNING_RATE_VALUES,
        MIN_CHILD_WEIGHT_VALUES,
    )

    return [
        {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_weight": (
                min_child_weight
            ),
        }
        for (
            max_depth,
            learning_rate,
            min_child_weight,
        ) in combinations
    ]


def main():
    arguments = parse_arguments()

    feature_path = FEATURE_DIRECTORY / (
        f"{arguments.model}.npz"
    )

    output_path = OUTPUT_DIRECTORY / (
        f"{arguments.model}_validation.json"
    )

    feature_splits = load_feature_splits(
        feature_path
    )

    X_train = feature_splits["train"]["X"]
    y_train_text = (
        feature_splits["train"]["y"]
    )

    X_validation = (
        feature_splits["validation"]["X"]
    )
    y_validation_text = (
        feature_splits["validation"]["y"]
    )

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(
        y_train_text
    )

    y_validation = label_encoder.transform(
        y_validation_text
    )

    if len(label_encoder.classes_) != 2:
        raise ValueError(
            "Eksperiment zahteva tačno "
            "dve klase."
        )

    parameter_combinations = (
        create_parameter_combinations()
    )

    print(
        "Eksperiment 1 – patch-level "
        "binarna klasifikacija"
    )
    print("Klasifikator: XGBoost")
    print(
        "Feature extractor:",
        arguments.model,
    )
    print("Uređaj:", arguments.device)
    print("Train oblik:", X_train.shape)
    print(
        "Validation oblik:",
        X_validation.shape,
    )
    print(
        "Klase:",
        label_encoder.classes_.tolist(),
    )
    print(
        "Broj konfiguracija:",
        len(parameter_combinations),
    )

    validation_results = []

    for configuration_index, parameters in (
        enumerate(
            parameter_combinations,
            start=1,
        )
    ):
        print(
            f"\nKonfiguracija "
            f"{configuration_index}/"
            f"{len(parameter_combinations)}"
        )

        print(
            f"max_depth="
            f"{parameters['max_depth']}, "
            f"learning_rate="
            f"{parameters['learning_rate']}, "
            f"min_child_weight="
            f"{parameters['min_child_weight']}",
            flush=True,
        )

        classifier = XGBClassifier(
            objective="binary:logistic",
            n_estimators=MAX_ESTIMATORS,
            learning_rate=parameters[
                "learning_rate"
            ],
            max_depth=parameters[
                "max_depth"
            ],
            min_child_weight=parameters[
                "min_child_weight"
            ],
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            tree_method="hist",
            device=arguments.device,
            eval_metric="logloss",
            early_stopping_rounds=(
                EARLY_STOPPING_ROUNDS
            ),
            random_state=42,
            verbosity=0,
        )

        start_time = time.perf_counter()

        classifier.fit(
            X_train,
            y_train,
            eval_set=[
                (
                    X_validation,
                    y_validation,
                )
            ],
            verbose=False,
        )

        training_time = (
            time.perf_counter()
            - start_time
        )

        validation_predictions_encoded = (
            classifier.predict(
                X_validation
            )
        )

        validation_predictions = (
            label_encoder.inverse_transform(
                validation_predictions_encoded
                .astype(np.int64)
            )
        )

        metrics = calculate_metrics(
            y_validation_text,
            validation_predictions,
        )

        best_iteration = (
            classifier.best_iteration
        )

        number_of_estimators = (
            int(best_iteration) + 1
            if best_iteration is not None
            else MAX_ESTIMATORS
        )

        result = {
            "max_depth": parameters[
                "max_depth"
            ],
            "learning_rate": parameters[
                "learning_rate"
            ],
            "min_child_weight": parameters[
                "min_child_weight"
            ],
            "maximum_estimators": (
                MAX_ESTIMATORS
            ),
            "best_iteration": (
                int(best_iteration)
                if best_iteration is not None
                else None
            ),
            "n_estimators": (
                number_of_estimators
            ),
            "training_time_seconds": float(
                training_time
            ),
            **metrics,
        }

        validation_results.append(result)

        print(
            f"accuracy="
            f"{metrics['accuracy']:.4f}, "
            f"balanced_accuracy="
            f"{metrics['balanced_accuracy']:.4f}, "
            f"macro_f1="
            f"{metrics['macro_f1']:.4f}, "
            f"defect_f1="
            f"{metrics['defect_f1']:.4f}, "
            f"best_iteration="
            f"{best_iteration}, "
            f"training_time="
            f"{training_time:.1f}s"
        )

    best_result = max(
        validation_results,
        key=lambda result: (
            result["macro_f1"]
        ),
    )

    experiment_result = {
        "experiment": (
            "experiment_1_patch_binary"
        ),
        "classifier": "xgboost",
        "feature_extractor": (
            arguments.model
        ),
        "device": arguments.device,
        "class_labels": (
            label_encoder.classes_.tolist()
        ),
        "selection_metric": "macro_f1",
        "best_parameters": {
            "max_depth": best_result[
                "max_depth"
            ],
            "learning_rate": best_result[
                "learning_rate"
            ],
            "min_child_weight": best_result[
                "min_child_weight"
            ],
            "n_estimators": best_result[
                "n_estimators"
            ],
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        "best_validation_macro_f1": (
            best_result["macro_f1"]
        ),
        "validation_results": (
            validation_results
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            experiment_result,
            output_file,
            indent=2,
        )

    print("\nNajbolji validation rezultat:")
    print(
        "max_depth:",
        best_result["max_depth"],
    )
    print(
        "learning_rate:",
        best_result["learning_rate"],
    )
    print(
        "min_child_weight:",
        best_result["min_child_weight"],
    )
    print(
        "n_estimators:",
        best_result["n_estimators"],
    )
    print(
        "Accuracy:",
        f"{best_result['accuracy']:.4f}",
    )
    print(
        "Macro F1:",
        f"{best_result['macro_f1']:.4f}",
    )
    print(
        "Defect F1:",
        f"{best_result['defect_f1']:.4f}",
    )
    print("Sačuvano:", output_path)


if __name__ == "__main__":
    main()