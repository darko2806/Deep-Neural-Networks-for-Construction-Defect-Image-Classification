from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter
import statistics

# ============================================================
# PATHS
# ============================================================

DATASET_DIR = Path("../data/MBDD2025")
ANNOTATIONS_DIR = DATASET_DIR / "Annotations"
IMAGES_DIR = DATASET_DIR / "JPEGImages"
LABELS_DIR = DATASET_DIR / "Labels"


# ============================================================
# STORAGE
# ============================================================

class_instance_counts = Counter()
class_image_counts = Counter()

objects_per_image = []
classes_per_image = []

class_combinations = Counter()

images_with_zero_objects = []
images_with_one_object = []
images_with_multiple_objects = []

images_with_one_class = []
images_with_multiple_classes = []

all_classes = set()

image_sizes = Counter()

xml_files = sorted(ANNOTATIONS_DIR.glob("*.xml"))


# ============================================================
# PARSE XML FILES
# ============================================================

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

    if size is not None:
        width = size.findtext("width")
        height = size.findtext("height")

        if width is not None and height is not None:
            image_sizes[(int(float(width)), int(float(height)))] += 1

    # --------------------------------------------------------
    # Objects / defects
    # --------------------------------------------------------

    objects = root.findall("object")

    object_classes = []

    for obj in objects:

        class_name = obj.findtext("name")

        if class_name is None:
            continue

        class_name = class_name.strip().lower()

        object_classes.append(class_name)

        all_classes.add(class_name)
        class_instance_counts[class_name] += 1

    # Number of bounding boxes
    n_objects = len(object_classes)
    objects_per_image.append(n_objects)

    if n_objects == 0:
        images_with_zero_objects.append(xml_path.stem)

    elif n_objects == 1:
        images_with_one_object.append(xml_path.stem)

    else:
        images_with_multiple_objects.append(xml_path.stem)

    # --------------------------------------------------------
    # Unique defect types in image
    # --------------------------------------------------------

    unique_classes = sorted(set(object_classes))
    n_classes = len(unique_classes)

    classes_per_image.append(n_classes)

    for class_name in unique_classes:
        class_image_counts[class_name] += 1

    if n_classes == 1:
        images_with_one_class.append(xml_path.stem)

    elif n_classes > 1:
        images_with_multiple_classes.append(xml_path.stem)

    # Combination of defect classes appearing in image
    if unique_classes:
        combination = tuple(unique_classes)
        class_combinations[combination] += 1


# ============================================================
# FILE CONSISTENCY
# ============================================================

image_stems = {p.stem for p in IMAGES_DIR.glob("*.jpg")}
xml_stems = {p.stem for p in ANNOTATIONS_DIR.glob("*.xml")}
txt_stems = {p.stem for p in LABELS_DIR.glob("*.txt")}

xml_without_image = xml_stems - image_stems
image_without_xml = image_stems - xml_stems

xml_without_txt = xml_stems - txt_stems
txt_without_xml = txt_stems - xml_stems


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MBDD2025 DATASET ANALYSIS")
print("=" * 70)

print("\n--- BASIC COUNTS ---")

print(f"XML annotations: {len(xml_stems)}")
print(f"JPG images:      {len(image_stems)}")
print(f"TXT labels:      {len(txt_stems)}")

print("\n--- CLASSES ---")

print(f"Number of classes: {len(all_classes)}")
print(f"Classes: {sorted(all_classes)}")

print("\n--- DEFECT INSTANCES PER CLASS ---")

for cls, count in class_instance_counts.most_common():
    print(f"{cls:20s}: {count}")

print("\n--- IMAGES CONTAINING EACH CLASS ---")

for cls, count in class_image_counts.most_common():
    percentage = 100 * count / len(xml_files)

    print(
        f"{cls:20s}: "
        f"{count:6d} images "
        f"({percentage:6.2f}%)"
    )

print("\n--- NUMBER OF OBJECTS / BOUNDING BOXES ---")

print(f"Images with 0 objects:  {len(images_with_zero_objects)}")
print(f"Images with 1 object:   {len(images_with_one_object)}")
print(f"Images with >1 objects: {len(images_with_multiple_objects)}")

if objects_per_image:

    print(
        f"Average objects/image: "
        f"{statistics.mean(objects_per_image):.3f}"
    )

    print(
        f"Median objects/image: "
        f"{statistics.median(objects_per_image):.3f}"
    )

    print(
        f"Maximum objects/image: "
        f"{max(objects_per_image)}"
    )

print("\n--- NUMBER OF UNIQUE DEFECT TYPES PER IMAGE ---")

type_count_distribution = Counter(classes_per_image)

for n_classes in sorted(type_count_distribution):

    count = type_count_distribution[n_classes]
    percentage = 100 * count / len(xml_files)

    print(
        f"{n_classes} unique class(es): "
        f"{count:6d} images "
        f"({percentage:6.2f}%)"
    )

print("\n--- SINGLE VS MULTIPLE DEFECT TYPES ---")

print(
    f"Images with exactly one defect type: "
    f"{len(images_with_one_class)}"
)

print(
    f"Images with multiple defect types:   "
    f"{len(images_with_multiple_classes)}"
)

print("\n--- CLASS COMBINATIONS ---")

for combination, count in class_combinations.most_common():

    combination_string = " + ".join(combination)

    print(
        f"{combination_string:50s}: "
        f"{count}"
    )

print("\n--- IMAGE RESOLUTIONS ---")

for resolution, count in image_sizes.most_common():

    print(
        f"{resolution[0]} x {resolution[1]}: "
        f"{count}"
    )

print("\n--- CONSISTENCY CHECK ---")

print(f"XML without JPG: {len(xml_without_image)}")
print(f"JPG without XML: {len(image_without_xml)}")

print(f"XML without TXT: {len(xml_without_txt)}")
print(f"TXT without XML: {len(txt_without_xml)}")

if xml_without_image:
    print("\nExamples XML without image:")
    print(sorted(xml_without_image)[:10])

if image_without_xml:
    print("\nExamples images without XML:")
    print(sorted(image_without_xml)[:10])

if images_with_zero_objects:
    print("\nExamples images with NO annotated objects:")
    print(images_with_zero_objects[:20])

if images_with_multiple_classes:
    print("\nExamples images with MULTIPLE defect types:")
    print(images_with_multiple_classes[:20])

print("\n" + "=" * 70)
print("END")
print("=" * 70)