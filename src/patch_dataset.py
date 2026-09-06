from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class PatchDefectDataset(Dataset):
    def __init__(
        self,
        metadata,
        transform=None,
    ):
        required_columns = {
            "patch_id",
            "image_id",
            "image_path",
            "target_class",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
        }

        missing_columns = (
            required_columns
            - set(metadata.columns)
        )

        if missing_columns:
            raise ValueError(
                "Nedostaju obavezne kolone: "
                f"{sorted(missing_columns)}"
            )

        if metadata[
            "target_class"
        ].isna().any():
            raise ValueError(
                "target_class ne sme sadržati "
                "nedostajuće vrednosti."
            )

        if metadata[
            "patch_id"
        ].duplicated().any():
            raise ValueError(
                "patch_id vrednosti moraju biti "
                "jedinstvene."
            )

        expected_classes = {
            "defect",
            "no_defect",
        }

        found_classes = set(
            metadata[
                "target_class"
            ].unique()
        )

        if found_classes != expected_classes:
            raise ValueError(
                "Očekivane klase su "
                f"{sorted(expected_classes)}, "
                "a pronađene su "
                f"{sorted(found_classes)}."
            )

        self.metadata = (
            metadata
            .reset_index(drop=True)
            .copy()
        )

        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]

        image_path = Path(
            row["image_path"]
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Slika ne postoji: {image_path}"
            )

        xmin = int(row["xmin"])
        ymin = int(row["ymin"])
        xmax = int(row["xmax"])
        ymax = int(row["ymax"])

        if (
            xmin < 0
            or ymin < 0
            or xmax <= xmin
            or ymax <= ymin
        ):
            raise ValueError(
                "Neispravne koordinate za "
                f"{row['patch_id']}: "
                f"{xmin}, {ymin}, {xmax}, {ymax}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if (
                xmax > image.width
                or ymax > image.height
            ):
                raise ValueError(
                    "Patch izlazi van slike za "
                    f"{row['patch_id']}."
                )

            patch = image.crop(
                (
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                )
            )

            if self.transform is not None:
                patch = self.transform(patch)
            else:
                patch = patch.copy()

        return {
            "image": patch,

            # Postojeća extract_features funkcija
            # očekuje ključ image_id. Ovde patch_id
            # predstavlja jedinstveni uzorak.
            "image_id": str(
                row["patch_id"]
            ),

            "target_class": str(
                row["target_class"]
            ),
        }