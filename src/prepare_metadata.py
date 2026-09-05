from pathlib import Path

from src.dataset import (
    add_content_hashes,
    assign_data_splits,
    build_metadata,
    select_single_class_images,
)


def main():
    dataset_dir = Path("MBDD2025")
    output_path = Path("data/metadata.csv")

    print("Čitanje XML anotacija...")
    metadata = build_metadata(dataset_dir)

    print("Izdvajanje jednoklasnih slika...")
    metadata = select_single_class_images(metadata)

    print("Računanje hash vrednosti slika...")
    metadata = add_content_hashes(metadata)

    print("Pravljenje train/validation/test podele...")
    metadata = assign_data_splits(metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_path, index=False)

    print(f"\nMetadata je sačuvan u: {output_path}")
    print(f"Broj redova: {len(metadata)}")
    print(f"Broj kolona: {len(metadata.columns)}")

    print("\nBroj slika po splitu:")
    print(metadata["split"].value_counts())


if __name__ == "__main__":
    main()