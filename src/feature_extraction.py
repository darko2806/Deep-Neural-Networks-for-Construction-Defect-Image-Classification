import timm
import torch
from timm.data import resolve_model_data_config
import numpy as np
from src.transforms import build_inference_transform


FEATURE_EXTRACTOR_MODELS = {
    "resnet50": "resnet50.tv_in1k",
    "inception_resnet_v2": "inception_resnet_v2.tf_in1k",
    "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in1k",
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def create_feature_extractor(model_key, device=None):
    if model_key not in FEATURE_EXTRACTOR_MODELS:
        raise ValueError(
            f"Nepoznat model: {model_key}. "
            f"Dostupni modeli: {sorted(FEATURE_EXTRACTOR_MODELS)}"
        )

    if device is None:
        device = get_device()

    model_name = FEATURE_EXTRACTOR_MODELS[model_key]

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,
    )

    for parameter in model.parameters():
        parameter.requires_grad = False

    model = model.to(device)
    model.eval()

    data_config = resolve_model_data_config(model)

    image_height = data_config["input_size"][1]
    image_width = data_config["input_size"][2]

    if image_height != image_width:
        raise ValueError(
            "ResizeWithPadding trenutno podržava samo kvadratne ulaze"
        )

    transform = build_inference_transform(
        image_size=image_height,
        mean=data_config["mean"],
        std=data_config["std"],
        interpolation=data_config["interpolation"],
    )

    return model, transform, device

def extract_features(model, data_loader, device):
    feature_batches = []
    image_ids = []
    target_classes = []

    total_batches = len(data_loader)

    with torch.inference_mode():
        for batch_index, batch in enumerate(data_loader, start=1):
            images = batch["image"].to(
                device,
                non_blocking=device.type == "cuda",
            )

            batch_features = model(images)

            if batch_features.ndim != 2:
                raise ValueError(
                    "Model mora vratiti matricu oblika "
                    "[batch_size, n_features]"
                )

            feature_batches.append(
                batch_features.float().cpu().numpy()
            )

            image_ids.extend(batch["image_id"])
            target_classes.extend(batch["target_class"])

            if batch_index % 50 == 0 or batch_index == total_batches:
                print(
                    f"Obrađen batch {batch_index}/{total_batches}"
                )

    features = np.concatenate(feature_batches, axis=0)

    extraction_result = {
        "features": features,
        "image_ids": np.asarray(image_ids),
        "target_classes": np.asarray(target_classes),
    }

    return extraction_result