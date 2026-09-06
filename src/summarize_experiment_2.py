import json
from pathlib import Path

import pandas as pd


RESULT_PATHS = [
    Path(
        "data/results/logistic_regression/"
        "resnet50_validation.json"
    ),
    Path(
        "data/results/logistic_regression/"
        "inception_resnet_v2_validation.json"
    ),
    Path(
        "data/results/logistic_regression/"
        "convnextv2_tiny_validation.json"
    ),
    Path(
        "data/results/random_forest/"
        "resnet50_validation.json"
    ),
    Path(
        "data/results/random_forest/"
        "inception_resnet_v2_validation.json"
    ),
    Path(
        "data/results/random_forest/"
        "convnextv2_tiny_validation.json"
    ),
    Path(
        "data/results/xgboost/"
        "resnet50_validation.json"
    ),
    Path(
        "data/results/xgboost/"
        "inception_resnet_v2_validation.json"
    ),
    Path(
        "data/results/xgboost/"
        "convnextv2_tiny_validation.json"
    ),
]

OUTPUT_DIRECTORY = Path(
    "data/results/experiment_2_multiclass"
)


def format_classifier_name(name):
    names = {
        "logistic_regression": "Logistic Regression",
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
        "convnextv2_tiny": "ConvNeXt V2 Tiny",
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
    ) as file:
        result = json.load(file)

    best_validation_result = max(
        result["validation_results"],
        key=lambda item: item["macro_f1"],
    )

    if result["classifier"] == "logistic_regression":
        best_parameters = {
            "C": result["best_C"],
        }
    else:
        best_parameters = result["best_parameters"]

    return {
        "classifier": format_classifier_name(
            result["classifier"]
        ),
        "feature_extractor": format_extractor_name(
            result["feature_extractor"]
        ),
        "accuracy": best_validation_result["accuracy"],
        "balanced_accuracy": best_validation_result[
            "balanced_accuracy"
        ],
        "macro_f1": best_validation_result["macro_f1"],
        "weighted_f1": best_validation_result[
            "weighted_f1"
        ],
        "best_parameters": json.dumps(
            best_parameters,
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def create_markdown_table(summary, metadata):
    split_counts = (
        metadata["split"]
        .value_counts()
        .to_dict()
    )

    lines = [
        "# Eksperiment 2 – multiclass klasifikacija tipa defekta",
        "",
        (
            "Korišćene su slike sa tačno jednom vrstom "
            "defekta, nakon perceptualnog grupisanja "
            "sa pHash pragom 6 i uklanjanja konfliktnih "
            "vizuelnih grupa."
        ),
        "",
        f"- Ukupan broj slika: {len(metadata)}",
        f"- Train: {split_counts.get('train', 0)}",
        (
            "- Validation: "
            f"{split_counts.get('validation', 0)}"
        ),
        f"- Test: {split_counts.get('test', 0)}",
        "- Metrika izbora: validation macro F1",
        "",
        (
            "| Rang | Feature extractor | Klasifikator | "
            "Accuracy | Balanced accuracy | Macro F1 | "
            "Weighted F1 |"
        ),
        (
            "|---:|---|---|---:|---:|---:|---:|"
        ),
    ]

    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.rank} "
            f"| {row.feature_extractor} "
            f"| {row.classifier} "
            f"| {row.accuracy:.4f} "
            f"| {row.balanced_accuracy:.4f} "
            f"| {row.macro_f1:.4f} "
            f"| {row.weighted_f1:.4f} |"
        )

    winner = summary.iloc[0]

    lines.extend(
        [
            "",
            "## Izabrani model",
            "",
            (
                f"**{winner['feature_extractor']} + "
                f"{winner['classifier']}**"
            ),
            "",
            (
                f"Validation macro F1: "
                f"**{winner['macro_f1']:.4f}**"
            ),
            "",
            "Najbolji hiperparametri:",
            "",
            f"`{winner['best_parameters']}`",
            "",
        ]
    )

    return "\n".join(lines)


def main():
    metadata = pd.read_csv(
        "data/metadata.csv"
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
        range(1, len(summary) + 1),
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        OUTPUT_DIRECTORY
        / "validation_model_comparison.csv"
    )
    markdown_path = (
        OUTPUT_DIRECTORY
        / "validation_model_comparison.md"
    )

    summary.to_csv(
        csv_path,
        index=False,
    )

    markdown_text = create_markdown_table(
        summary,
        metadata,
    )

    markdown_path.write_text(
        markdown_text,
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print(f"\nCSV tabela: {csv_path}")
    print(f"Markdown izveštaj: {markdown_path}")


if __name__ == "__main__":
    main()