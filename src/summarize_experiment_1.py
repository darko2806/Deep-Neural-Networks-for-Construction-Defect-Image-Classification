import json
from pathlib import Path

import pandas as pd


RESULT_DIRECTORY = Path(
    "data/results/experiment_1_binary"
)

RESULT_PATHS = [
    RESULT_DIRECTORY
    / "logistic_regression"
    / "resnet50_validation.json",

    RESULT_DIRECTORY
    / "logistic_regression"
    / "inception_resnet_v2_validation.json",

    RESULT_DIRECTORY
    / "logistic_regression"
    / "convnextv2_tiny_validation.json",

    RESULT_DIRECTORY
    / "random_forest"
    / "resnet50_validation.json",

    RESULT_DIRECTORY
    / "random_forest"
    / "inception_resnet_v2_validation.json",

    RESULT_DIRECTORY
    / "random_forest"
    / "convnextv2_tiny_validation.json",

    RESULT_DIRECTORY
    / "xgboost"
    / "resnet50_validation.json",

    RESULT_DIRECTORY
    / "xgboost"
    / "inception_resnet_v2_validation.json",

    RESULT_DIRECTORY
    / "xgboost"
    / "convnextv2_tiny_validation.json",
]

METADATA_PATH = Path(
    "data/patch_metadata.csv"
)


def format_classifier_name(name):
    names = {
        "logistic_regression": (
            "Logistic Regression"
        ),
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }

    return names[name]


def format_extractor_name(name):
    names = {
        "resnet50": "ResNet50",
        "inception_resnet_v2": (
            "Inception-ResNet-v2"
        ),
        "convnextv2_tiny": (
            "ConvNeXt V2 Tiny"
        ),
    }

    return names[name]


def load_result(result_path):
    if not result_path.is_file():
        raise FileNotFoundError(
            f"Nedostaje rezultat: {result_path}"
        )

    with result_path.open(
        "r",
        encoding="utf-8",
    ) as result_file:
        result = json.load(result_file)

    best_validation_result = max(
        result["validation_results"],
        key=lambda row: row["macro_f1"],
    )

    classifier_name = result["classifier"]

    if classifier_name == (
        "logistic_regression"
    ):
        best_parameters = {
            "C": result["best_C"],
        }
    else:
        best_parameters = result[
            "best_parameters"
        ]

    return {
        "classifier": (
            format_classifier_name(
                classifier_name
            )
        ),
        "feature_extractor": (
            format_extractor_name(
                result["feature_extractor"]
            )
        ),
        "accuracy": (
            best_validation_result[
                "accuracy"
            ]
        ),
        "balanced_accuracy": (
            best_validation_result[
                "balanced_accuracy"
            ]
        ),
        "macro_f1": (
            best_validation_result[
                "macro_f1"
            ]
        ),
        "weighted_f1": (
            best_validation_result[
                "weighted_f1"
            ]
        ),
        "defect_f1": (
            best_validation_result[
                "defect_f1"
            ]
        ),
        "best_parameters": json.dumps(
            best_parameters,
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def create_markdown_report(
    summary,
    metadata,
):
    split_counts = (
        metadata["split"]
        .value_counts()
        .to_dict()
    )

    target_counts = (
        metadata["target_class"]
        .value_counts()
        .to_dict()
    )

    number_of_pairs = int(
        metadata["pair_id"].nunique()
    )

    number_of_images = int(
        metadata["image_id"].nunique()
    )

    number_of_visual_groups = int(
        metadata["visual_group"].nunique()
    )

    lines = [
        (
            "# Eksperiment 1 – patch-level "
            "binarna klasifikacija"
        ),
        "",
        (
            "Zadatak je klasifikacija patch-eva "
            "u klase `defect` i `no_defect`. "
            "Pozitivni patch-evi formirani su oko "
            "anotiranih defekata, dok su negativni "
            "patch-evi izabrani iz delova iste slike "
            "koji ne presecaju anotirane regione."
        ),
        "",
        (
            "Ovaj eksperiment je eksploratoran. "
            "Oznaka `no_defect` znači da patch ne "
            "preseca poznatu anotaciju; ona ne "
            "garantuje da patch predstavlja potpuno "
            "zdravu građevinsku površinu."
        ),
        "",
        f"- Broj originalnih slika: {number_of_images}",
        (
            "- Broj vizuelnih grupa: "
            f"{number_of_visual_groups}"
        ),
        f"- Broj patch parova: {number_of_pairs}",
        f"- Ukupan broj patch-eva: {len(metadata)}",
        (
            "- Defect patch-evi: "
            f"{target_counts.get('defect', 0)}"
        ),
        (
            "- No-defect patch-evi: "
            f"{target_counts.get('no_defect', 0)}"
        ),
        (
            "- Train: "
            f"{split_counts.get('train', 0)}"
        ),
        (
            "- Validation: "
            f"{split_counts.get('validation', 0)}"
        ),
        (
            "- Test: "
            f"{split_counts.get('test', 0)}"
        ),
        (
            "- Metrika izbora modela: "
            "validation Macro F1"
        ),
        "",
        (
            "| Rang | Feature extractor | "
            "Klasifikator | Accuracy | "
            "Balanced accuracy | Macro F1 | "
            "Weighted F1 | Defect F1 |"
        ),
        (
            "|---:|---|---|---:|---:|---:|"
            "---:|---:|"
        ),
    ]

    for row in summary.itertuples(
        index=False
    ):
        lines.append(
            f"| {row.rank} "
            f"| {row.feature_extractor} "
            f"| {row.classifier} "
            f"| {row.accuracy:.4f} "
            f"| {row.balanced_accuracy:.4f} "
            f"| {row.macro_f1:.4f} "
            f"| {row.weighted_f1:.4f} "
            f"| {row.defect_f1:.4f} |"
        )

    winner = summary.iloc[0]

    lines.extend(
        [
            "",
            "## Izabrani model",
            "",
            (
                f"**{winner['feature_extractor']} "
                f"+ {winner['classifier']}**"
            ),
            "",
            (
                "Validation Macro F1: "
                f"**{winner['macro_f1']:.4f}**"
            ),
            "",
            (
                "Validation Defect F1: "
                f"**{winner['defect_f1']:.4f}**"
            ),
            "",
            "Najbolji hiperparametri:",
            "",
            f"`{winner['best_parameters']}`",
            "",
            (
                "Model je izabran bez korišćenja "
                "test skupa."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def main():
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Nedostaje metadata: {METADATA_PATH}"
        )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    rows = [
        load_result(result_path)
        for result_path in RESULT_PATHS
    ]

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        by="macro_f1",
        ascending=False,
    ).reset_index(drop=True)

    summary.insert(
        0,
        "rank",
        range(
            1,
            len(summary) + 1,
        ),
    )

    RESULT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = RESULT_DIRECTORY / (
        "validation_model_comparison.csv"
    )

    markdown_path = RESULT_DIRECTORY / (
        "validation_model_comparison.md"
    )

    summary.to_csv(
        csv_path,
        index=False,
    )

    markdown_report = (
        create_markdown_report(
            summary,
            metadata,
        )
    )

    markdown_path.write_text(
        markdown_report,
        encoding="utf-8",
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print("\nCSV tabela:", csv_path)
    print(
        "Markdown izveštaj:",
        markdown_path,
    )


if __name__ == "__main__":
    main()