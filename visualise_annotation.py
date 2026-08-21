import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Promeni ove putanje
image_path = "./MBDD2025/JPEGImages/Hefei118.jpg"
xml_path = "./MBDD2025/Annotations/Hefei118.xml"

# Ucitaj sliku
image = Image.open(image_path).convert("RGB")

# Procitaj XML
tree = ET.parse(xml_path)
root = tree.getroot()

# Napravi objekat za crtanje po slici
draw = ImageDraw.Draw(image)

# Jedna slika moze imati vise bounding box-ova
for obj in root.findall("object"):

    # Klasa objekta
    class_name = obj.find("name").text

    # Bounding box
    bbox = obj.find("bndbox")

    xmin = int(float(bbox.find("xmin").text))
    ymin = int(float(bbox.find("ymin").text))
    xmax = int(float(bbox.find("xmax").text))
    ymax = int(float(bbox.find("ymax").text))

    # Nacrtaj pravougaonik
    draw.rectangle(
        [(xmin, ymin), (xmax, ymax)],
        outline="red",
        width=4
    )

    # Napiši klasu
    draw.text(
        (xmin, ymin - 15),
        class_name,
        fill="red"
    )

# Prikazi rezultat
plt.figure(figsize=(12, 8))
plt.imshow(image)
plt.axis("off")
plt.show()