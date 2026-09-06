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
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.feature_data import load_feature_splits
from src.feature_extraction import FEATURE_EXTRACTOR_MODELS


MAX_DEPTH_VALUES = [4, 8]
LEARNING_RATE_VALUES = [0.03, 0.1]
BALANCED_WEIGHT_VALUES = [False, True]

MAX_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 50


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Izbor XGBoost hiperparametara "
            "na validation skupu"
        )
    )

    parser.add_argument(
        "--model",
        choices=sorted(FEATURE_EXTRACTOR_MODELS),
        required=True,
    )

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )

    return parser.parse_args()


def calculate_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }

    return {
        name: float(value)
        for name, value in metrics.items()
    }


def create_parameter_combinations():
    combinations = product(
        MAX_DEPTH_VALUES,
        LEARNING_RATE_VALUES,
        BALANCED_WEIGHT_VALUES,
    )

    return [
        {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "balanced_weights": balanced_weights,
        }
        for (
            max_depth,
            learning_rate,
            balanced_weights,
        ) in combinations
    ]


def main():
    args = parse_arguments()

    feature_path = Path(
        f"data/features/{args.model}.npz"
    )

    output_path = Path(
        "data/results/xgboost/"
        f"{args.model}_validation.json"
    )

    feature_splits = load_feature_splits(
        feature_path
    )

    X_train = feature_splits["train"]["X"]
    y_train_text = feature_splits["train"]["y"]

    X_validation = feature_splits[
        "validation"
    ]["X"]
    y_validation_text = feature_splits[
        "validation"
    ]["y"]

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(
        y_train_text
    )
    y_validation = label_encoder.transform(
        y_validation_text
    )

    balanced_sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    ).astype(np.float32)

    parameter_combinations = (
        create_parameter_combinations()
    )

    validation_results = []

    print(f"Feature extractor: {args.model}")
    print(f"Uređaj: {args.device}")
    print(f"Train oblik: {X_train.shape}")
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

    for configuration_index, parameters in enumerate(
        parameter_combinations,
        start=1,
    ):
        print(
            f"\nKonfiguracija "
            f"{configuration_index}/"
            f"{len(parameter_combinations)}"
        )
        print(
            f"max_depth={parameters['max_depth']}, "
            f"learning_rate="
            f"{parameters['learning_rate']}, "
            f"balanced_weights="
            f"{parameters['balanced_weights']}",
            flush=True,
        )

        classifier = XGBClassifier(
            objective="multi:softprob",
            num_class=len(label_encoder.classes_),
            n_estimators=MAX_ESTIMATORS,
            learning_rate=parameters[
                "learning_rate"
            ],
            max_depth=parameters["max_depth"],
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            tree_method="hist",
            device=args.device,
            eval_metric="mlogloss",
            early_stopping_rounds=(
                EARLY_STOPPING_ROUNDS
            ),
            random_state=42,
            verbosity=0,
        )

        fit_arguments = {
            "X": X_train,
            "y": y_train,
            "eval_set": [
                (
                    X_validation,
                    y_validation,
                )
            ],
            "verbose": False,
        }

        if parameters["balanced_weights"]:
            fit_arguments["sample_weight"] = (
                balanced_sample_weights
            )

        training_start = time.perf_counter()

        classifier.fit(**fit_arguments)

        training_time = (
            time.perf_counter()
            - training_start
        )

        prediction_start = time.perf_counter()

        validation_predictions_encoded = (
            classifier.predict(X_validation)
        )

        prediction_time = (
            time.perf_counter()
            - prediction_start
        )

        validation_predictions = (
            label_encoder.inverse_transform(
                validation_predictions_encoded.astype(
                    np.int64
                )
            )
        )

        metrics = calculate_metrics(
            y_validation_text,
            validation_predictions,
        )

        best_iteration = classifier.best_iteration

        result = {
            "max_depth": parameters["max_depth"],
            "learning_rate": parameters[
                "learning_rate"
            ],
            "balanced_weights": parameters[
                "balanced_weights"
            ],
            "maximum_estimators": MAX_ESTIMATORS,
            "best_iteration": (
                int(best_iteration)
                if best_iteration is not None
                else None
            ),
            "training_time_seconds": float(
                training_time
            ),
            "prediction_time_seconds": float(
                prediction_time
            ),
            **metrics,
        }

        validation_results.append(result)

        print(
            f"accuracy={metrics['accuracy']:.4f}, "
            f"balanced_accuracy="
            f"{metrics['balanced_accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}, "
            f"weighted_f1="
            f"{metrics['weighted_f1']:.4f}, "
            f"best_iteration={best_iteration}, "
            f"training_time={training_time:.1f}s"
        )

    best_result = max(
        validation_results,
        key=lambda result: result["macro_f1"],
    )

    experiment_result = {
        "classifier": "xgboost",
        "feature_extractor": args.model,
        "device": args.device,
        "class_labels": (
            label_encoder.classes_.tolist()
        ),
        "selection_metric": "macro_f1",
        "best_parameters": {
            "max_depth": best_result["max_depth"],
            "learning_rate": best_result[
                "learning_rate"
            ],
            "balanced_weights": best_result[
                "balanced_weights"
            ],
            "n_estimators": (
                best_result["best_iteration"] + 1
            ),
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        "best_validation_macro_f1": (
            best_result["macro_f1"]
        ),
        "validation_results": validation_results,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            experiment_result,
            file,
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
        "balanced_weights:",
        best_result["balanced_weights"],
    )
    print(
        "n_estimators:",
        best_result["best_iteration"] + 1,
    )
    print(
        "Macro F1:",
        f"{best_result['macro_f1']:.4f}",
    )
    print(
        "Rezultati su sačuvani u:",
        output_path,
    )


if __name__ == "__main__":
    main()