from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


# ============================================================
# PATH
# ============================================================

DATASET_DIR = Path("../data/MBDD2025")
ANNOTATIONS_DIR = DATASET_DIR / "Annotations"

# Input sizes that are relevant for our CNN feature extractors
CNN_INPUT_SIZES = [224, 299]


# ============================================================
# STORAGE
# ============================================================

bbox_records = []
image_records = []


# ============================================================
# PARSE ALL XML FILES
# ============================================================

xml_files = sorted(ANNOTATIONS_DIR.glob("*.xml"))

for xml_path in xml_files:

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"ERROR reading {xml_path}: {e}")
        continue

    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    size = root.find("size")

    if size is None:
        continue

    width = float(size.findtext("width"))
    height = float(size.findtext("height"))

    image_area = width * height

    objects = root.findall("object")

    image_bbox_areas = []

    for obj in objects:

        class_name = obj.findtext("name")

        if class_name is None:
            continue

        class_name = class_name.strip().lower()

        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        xmin = float(bbox.findtext("xmin"))
        ymin = float(bbox.findtext("ymin"))
        xmax = float(bbox.findtext("xmax"))
        ymax = float(bbox.findtext("ymax"))

        # Bounding-box dimensions
        bbox_width = max(0.0, xmax - xmin)
        bbox_height = max(0.0, ymax - ymin)
        bbox_area = bbox_width * bbox_height

        if bbox_area <= 0:
            continue

        # ----------------------------------------------------
        # Relative dimensions
        # ----------------------------------------------------

        relative_width = bbox_width / width
        relative_height = bbox_height / height
        relative_area = bbox_area / image_area

        image_bbox_areas.append(relative_area)

        record = {
            "image": xml_path.stem,
            "class": class_name,

            "image_width": width,
            "image_height": height,

            "bbox_width": bbox_width,
            "bbox_height": bbox_height,
            "bbox_area": bbox_area,

            "relative_width": relative_width,
            "relative_height": relative_height,
            "relative_area": relative_area,
        }

        # ----------------------------------------------------
        # How large would this object be after resizing?
        # ----------------------------------------------------

        for input_size in CNN_INPUT_SIZES:

            record[f"width_at_{input_size}"] = (
                relative_width * input_size
            )

            record[f"height_at_{input_size}"] = (
                relative_height * input_size
            )

            record[f"area_at_{input_size}"] = (
                relative_area * input_size * input_size
            )

        bbox_records.append(record)

    # --------------------------------------------------------
    # Image-level information
    # --------------------------------------------------------

    if image_bbox_areas:

        image_records.append(
            {
                "image": xml_path.stem,
                "num_boxes": len(image_bbox_areas),

                # Largest individual defect
                "max_bbox_area_ratio": max(image_bbox_areas),

                # Smallest individual defect
                "min_bbox_area_ratio": min(image_bbox_areas),

                # Mean bounding-box area
                "mean_bbox_area_ratio": np.mean(image_bbox_areas),

                # NOTE:
                # This can overestimate actual covered area if
                # bounding boxes overlap.
                "sum_bbox_area_ratio": sum(image_bbox_areas),
            }
        )

    else:

        image_records.append(
            {
                "image": xml_path.stem,
                "num_boxes": 0,
                "max_bbox_area_ratio": 0,
                "min_bbox_area_ratio": 0,
                "mean_bbox_area_ratio": 0,
                "sum_bbox_area_ratio": 0,
            }
        )


# ============================================================
# DATAFRAMES
# ============================================================

bbox_df = pd.DataFrame(bbox_records)
image_df = pd.DataFrame(image_records)


# ============================================================
# SAVE RAW RESULTS
# ============================================================

bbox_df.to_csv("bbox_statistics.csv", index=False)
image_df.to_csv("image_bbox_statistics.csv", index=False)


# ============================================================
# HELPER FUNCTION
# ============================================================

def print_percentiles(values):

    values = np.asarray(values)

    percentiles = [
        1,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
    ]

    results = np.percentile(values, percentiles)

    for p, value in zip(percentiles, results):
        print(f"P{p:02d}: {value:.6f}")


# ============================================================
# GLOBAL STATISTICS
# ============================================================

print("\n" + "=" * 75)
print("MBDD2025 BOUNDING-BOX SIZE ANALYSIS")
print("=" * 75)

print("\n--- BASIC COUNTS ---")

print(f"Number of images:          {len(image_df)}")
print(f"Number of bounding boxes:  {len(bbox_df)}")

print("\n--- BOUNDING BOX AREA / IMAGE AREA ---")

print_percentiles(bbox_df["relative_area"])

print("\nMedian relative area:")
print(
    f"{100 * bbox_df['relative_area'].median():.3f}% "
    "of the entire image"
)


# ============================================================
# SMALL OBJECT ANALYSIS
# ============================================================

print("\n--- FRACTION OF SMALL DEFECTS ---")

thresholds = [
    0.001,   # 0.1%
    0.0025,  # 0.25%
    0.005,   # 0.5%
    0.01,    # 1%
    0.02,    # 2%
    0.05,    # 5%
    0.10,    # 10%
]

for threshold in thresholds:

    count = (bbox_df["relative_area"] < threshold).sum()
    percentage = 100 * count / len(bbox_df)

    print(
        f"Area < {100 * threshold:5.2f}% of image: "
        f"{count:6d} boxes ({percentage:6.2f}%)"
    )


# ============================================================
# EFFECT OF CNN RESIZING
# ============================================================

for input_size in CNN_INPUT_SIZES:

    print("\n" + "-" * 75)
    print(f"AFTER RESIZING FULL IMAGE TO {input_size} x {input_size}")
    print("-" * 75)

    width_column = f"width_at_{input_size}"
    height_column = f"height_at_{input_size}"

    print("\nBounding-box width in CNN input:")

    print(
        f"Median: "
        f"{bbox_df[width_column].median():.2f} pixels"
    )

    print(
        f"P10:    "
        f"{np.percentile(bbox_df[width_column], 10):.2f} pixels"
    )

    print(
        f"P90:    "
        f"{np.percentile(bbox_df[width_column], 90):.2f} pixels"
    )

    print("\nBounding-box height in CNN input:")

    print(
        f"Median: "
        f"{bbox_df[height_column].median():.2f} pixels"
    )

    print(
        f"P10:    "
        f"{np.percentile(bbox_df[height_column], 10):.2f} pixels"
    )

    print(
        f"P90:    "
        f"{np.percentile(bbox_df[height_column], 90):.2f} pixels"
    )

    # --------------------------------------------------------
    # Defects extremely small after resize
    # --------------------------------------------------------

    for pixel_threshold in [4, 8, 16, 32]:

        small = (
            (bbox_df[width_column] < pixel_threshold)
            | (bbox_df[height_column] < pixel_threshold)
        )

        count = small.sum()
        percentage = 100 * count / len(bbox_df)

        print(
            f"Width OR height < {pixel_threshold:2d}px: "
            f"{count:6d} "
            f"({percentage:6.2f}%)"
        )


# ============================================================
# PER-CLASS ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("PER-CLASS BOUNDING-BOX STATISTICS")
print("=" * 75)

classes = sorted(bbox_df["class"].unique())

for cls in classes:

    df_class = bbox_df[bbox_df["class"] == cls]

    print("\n" + "-" * 75)
    print(cls.upper())
    print("-" * 75)

    print(f"Number of instances: {len(df_class)}")

    print(
        f"Median area: "
        f"{100 * df_class['relative_area'].median():.3f}%"
    )

    print(
        f"Mean area:   "
        f"{100 * df_class['relative_area'].mean():.3f}%"
    )

    print(
        f"P10 area:    "
        f"{100 * np.percentile(df_class['relative_area'], 10):.3f}%"
    )

    print(
        f"P90 area:    "
        f"{100 * np.percentile(df_class['relative_area'], 90):.3f}%"
    )

    print("\nAt 224 x 224:")

    print(
        f"Median width:  "
        f"{df_class['width_at_224'].median():.2f}px"
    )

    print(
        f"Median height: "
        f"{df_class['height_at_224'].median():.2f}px"
    )


# ============================================================
# IMAGE-LEVEL ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("IMAGE-LEVEL DEFECT SIZE")
print("=" * 75)

images_with_boxes = image_df[image_df["num_boxes"] > 0]

print("\nLargest defect per image:")

print("Relative area percentiles:")

print_percentiles(
    images_with_boxes["max_bbox_area_ratio"]
)

print(
    "\nMedian largest defect occupies "
    f"{100 * images_with_boxes['max_bbox_area_ratio'].median():.3f}% "
    "of the image."
)

print("\nSum of bounding-box areas per image:")
print("(Can overestimate coverage when boxes overlap.)")

print_percentiles(
    images_with_boxes["sum_bbox_area_ratio"]
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("CLASS COUNTS VALIDATION")
print("=" * 75)

print(
    bbox_df["class"]
    .value_counts()
    .to_string()
)

print("\nCSV files created:")
print("  bbox_statistics.csv")
print("  image_bbox_statistics.csv")

print("\n" + "=" * 75)
print("END")
print("=" * 75)