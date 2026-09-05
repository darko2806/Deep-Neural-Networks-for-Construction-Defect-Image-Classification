from PIL import Image, ImageOps
from torchvision.transforms import Compose, Normalize, ToTensor


INTERPOLATION_METHODS = {
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


class ResizeWithPadding:
    def __init__(
        self,
        size,
        fill=(0, 0, 0),
        interpolation="bicubic",
    ):
        if interpolation not in INTERPOLATION_METHODS:
            raise ValueError(
                f"Nepoznata interpolacija: {interpolation}"
            )

        self.size = size
        self.fill = fill
        self.interpolation = INTERPOLATION_METHODS[interpolation]

    def __call__(self, image):
        resized_image = ImageOps.contain(
            image,
            (self.size, self.size),
            method=self.interpolation,
        )

        canvas = Image.new(
            mode="RGB",
            size=(self.size, self.size),
            color=self.fill,
        )

        left = (self.size - resized_image.width) // 2
        top = (self.size - resized_image.height) // 2

        canvas.paste(resized_image, (left, top))

        return canvas


def build_inference_transform(
    image_size,
    mean,
    std,
    interpolation="bicubic",
):
    padding_color = tuple(
        round(channel_mean * 255)
        for channel_mean in mean
    )

    transform = Compose(
        [
            ResizeWithPadding(
                size=image_size,
                fill=padding_color,
                interpolation=interpolation,
            ),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )

    return transform