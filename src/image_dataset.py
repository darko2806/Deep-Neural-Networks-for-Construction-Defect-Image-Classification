from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class DefectImageDataset(Dataset):
    def __init__(self, metadata, transform=None):
        required_columns = {
            "image_id",
            "image_path",
            "target_class",
        }

        missing_columns = required_columns - set(metadata.columns)

        if missing_columns:
            raise ValueError(
                f"Nedostaju obavezne kolone: {sorted(missing_columns)}"
            )

        if metadata["target_class"].isna().any():
            raise ValueError(
                "target_class ne sme sadržati nedostajuće vrednosti"
            )

        self.metadata = metadata.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        image_path = Path(row["image_path"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)
            else:
                image = image.copy()

        sample = {
            "image": image,
            "target_class": row["target_class"],
            "image_id": row["image_id"],
        }

        return sample