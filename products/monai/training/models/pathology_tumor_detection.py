# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import torch
import time
from torch.optim.lr_scheduler import StepLR

from monai.data import CSVDataset, DataLoader, PatchWSIDataset
from monai.networks.nets import TorchVisionFCModel
from monai.transforms import (
    Compose, Lambdad, GridSplitd, ToTensord, ToCupy, RandCuCIM,
    CuCIM, ToTensor, TorchVisiond, ToNumpyd, RandFlipd, RandRotate90d,
    CastToTyped, RandZoomd, ScaleIntensityRanged
)

# Preprocessing and location info
IMAGE_ROOT = "sample_images/camelyon/"
TRAIN_CSV = "demo/data/pathology_tumor_detection/training.csv"
VALID_CSV = "demo/data/pathology_tumor_detection/validation.csv"
REGION_SIZE = 256 * 3

def prepare_training_data(params):
    """
    Load, preprocess, and split the Camelyon16 dataset subset for the demo.

    Returns:
        dict with keys 'train' and 'val', each a DataLoader.
    """
    backend = params.get("backend", "cucim")

    # CPU/GPU support
    device = torch.device("cuda" if torch.cuda.is_available() and params.get("gpu", 0) >= 0 else "cpu")

    # Preprocessing for image and labelon both train and val
    # Only relevant steps for this binary patch classification
    preprocess_cpu_train = Compose([
        Lambdad(keys="label", func=lambda x: x.reshape((1, params["grid_shape"], params["grid_shape"]))),
        GridSplitd(
            keys=("image", "label"),
            grid=(params["grid_shape"], params["grid_shape"]),
            size={"image": params["patch_size"], "label": 1},
        ),
        ToTensord(keys="label"),  # image: handled downstream (to Tensor/CuPy)
    ])
    preprocess_cpu_valid = preprocess_cpu_train  # no random aug for this simple demo

    # Use minimal, safe augmentations (optional: add more as needed)
    # On GPU/CPU as required
    pre_transforms_gpu = Compose([
        ToCupy(),
        RandCuCIM(
            name="rand_color_jitter",
            prob=1.0,
            brightness=64.0 / 255.0,
            contrast=0.75,
            saturation=0.25,
            hue=0.04,
        ),
        RandCuCIM(name="rand_image_flip", prob=params["prob"], spatial_axis=-1),
        RandCuCIM(name="rand_image_rotate_90", prob=params["prob"], max_k=3, spatial_axis=(-2, -1)),
        CuCIM(name="scale_intensity_range", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
        ToTensor(device=device),
    ]) if backend == "cucim" else Compose([
        # Numpy/tensor pipeline for fallback
        TorchVisiond(
            keys="image", name="ColorJitter", brightness=64.0 / 255.0, contrast=0.75, saturation=0.25, hue=0.04
        ),
        ToNumpyd(keys="image"),
        RandFlipd(keys="image", prob=params["prob"], spatial_axis=-1),
        RandRotate90d(keys="image", prob=params["prob"]),
        CastToTyped(keys="image", dtype=np.float32),
        ScaleIntensityRanged(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
        ToTensord(keys="image"),
    ])

    preprocess_gpu_valid = Compose([
        ToCupy(dtype=np.float32),
        CuCIM(name="scale_intensity_range", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
        ToTensor(device=device),
    ]) if backend == "cucim" else Compose([
        CastToTyped(keys="image", dtype=np.float32),
        ScaleIntensityRanged(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
        ToTensord(keys="image"),
    ])

    # -- Setup datasets and loaders
    train_data_list = CSVDataset(
        TRAIN_CSV,
        col_groups={"image": 0, "location": [2, 1], "label": [3, 6, 9, 4, 7, 10, 5, 8, 11]},
        kwargs_read_csv={"header": None},
        transform=Lambdad("image", lambda x: os.path.join(IMAGE_ROOT, x + ".tif")),
    )
    train_dataset = PatchWSIDataset(
        data=train_data_list,
        patch_size=REGION_SIZE,
        patch_level=0,
        transform=preprocess_cpu_train,
        reader="openslide" if params.get("use_openslide", False) else "cuCIM",
    )
    train_loader = DataLoader(
        train_dataset, num_workers=0, batch_size=params["batch_size"], pin_memory=True
    )

    val_loader = None
    if not params.get("no_validate", False):
        valid_data_list = CSVDataset(
            VALID_CSV,
            col_groups={"image": 0, "location": [2, 1], "label": [3, 6, 9, 4, 7, 10, 5, 8, 11]},
            kwargs_read_csv={"header": None},
            transform=Lambdad("image", lambda x: os.path.join(IMAGE_ROOT, x + ".tif")),
        )
        valid_dataset = PatchWSIDataset(
            data=valid_data_list,
            patch_size=REGION_SIZE,
            patch_level=0,
            transform=preprocess_cpu_valid,
            reader="openslide" if params.get("use_openslide", False) else "cuCIM",
        )
        val_loader = DataLoader(
            valid_dataset, num_workers=0, batch_size=params["batch_size"], pin_memory=True
        )
    return {"train": train_loader, "val": val_loader}


def get_bundle_components(params):
    """
    Return MONAI-style preprocessing, postprocessing, and inferer for bundle export.
    """
    import monai
    from monai.inferers import SimpleInferer
    from monai.transforms import Compose, ToTensord, ScaleIntensityRanged

    # Minimal transforms for demo/test-only inference
    pre_transforms = Compose([
        ScaleIntensityRanged(
            keys=["image"], a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0, clip=True
        ),
        ToTensord(keys=["image"]),
    ])

    # Postprocessing: sigmoid then threshold to obtain class prediction (for BCE/logits)
    post_transforms = Compose([
        monai.transforms.Activations(sigmoid=True),
        monai.transforms.AsDiscrete(threshold=0.5)
    ])

    inferer = SimpleInferer()
    return pre_transforms, post_transforms, inferer


def extract_visualization_samples_from_loader(loader, n=10, with_label=True):
    """
    Show random patch tiles for dashboard visualization.
    """
    loader_iter = iter(loader)
    images = []
    for _ in range(n):
        try:
            batch = next(loader_iter)
            img = batch["image"]
            # Transpose to display as HWC RGB
            input_img = img[0].cpu().numpy()
            if input_img.ndim == 3 and input_img.shape[0] in [1, 3, 4]:
                input_img = np.transpose(input_img, (1, 2, 0))
            sample = {"input": input_img}
            if with_label and "label" in batch:
                label = batch["label"]
                label_img = label[0].cpu().numpy()
                if label_img.ndim == 3 and label_img.shape[0] == 1:
                    label_img = label_img[0]
                sample["label"] = label_img
            images.append(sample)
        except StopIteration:
            break
    return images

def identify_training_device(params):
    if torch.cuda.is_available() and params.get("gpu", 0) >= 0:
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def define_training_network(params, device):
    # Simple binary classifier based on ResNet; works for demonstration
    model = TorchVisionFCModel("resnet18", num_classes=1, use_conv=True, pretrained=params.get("pretrain", False))
    return model.to(device)

def define_loss_function(params):
    return torch.nn.BCEWithLogitsLoss()

def define_training_optimizer(params, model_parameters):
    # Use SGD for simplicity; switch as needed
    return torch.optim.SGD(model_parameters, lr=params.get("learning_rate", 1e-4), momentum=0.9)

def define_training_metric(params):
    # Compute accuracy across all samples in batch
    def accuracy_fn(y_pred, y_true):
        pred = (torch.sigmoid(y_pred) > 0.5).float()
        return (pred == y_true).float().mean().item()
    return accuracy_fn

def define_training_scheduler(params, optimizer):
    # You can expose these as params if you want to be configurable by
    # the UI
    step_size = params.get('step_size', 10)
    gamma = params.get('gamma', 0.1)
    lr_scheduler = StepLR(
            optimizer, 
            step_size=step_size, 
            gamma=gamma
    )
    return lr_scheduler

def train_batch(network, batch_data, loss_fn, optimizer, metric_fn, device, params, epoch):
    """
    Standardized demonstration batch-train step. Returns all expected metrics.
    """
    network.train()
    x = batch_data["image"].to(device).float()
    y = batch_data["label"].to(device).float()
    optimizer.zero_grad()
    outputs = network(x)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        # Predicted probability → binary
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        y_true = y
        # Metrics
        tp = (preds * y_true).sum().item()
        tn = ((1 - preds) * (1 - y_true)).sum().item()
        fp = (preds * (1 - y_true)).sum().item()
        fn = ((1 - preds) * y_true).sum().item()

        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
        precision = tp / (tp + fp + 1e-7) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn + 1e-7) if (tp + fn) > 0 else 0
        # Dice coefficient for binary (as mean_dice)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-7) if (2 * tp + fp + fn) > 0 else None
        per_class_dice = [dice]  # Only one class; format for compatibility

    return (
        loss.item(),
        dice,
        per_class_dice,
        accuracy,
        precision,
        recall,
    )

def evaluate_batch(network, batch_data, loss_fn, metric_fn, device, params, epoch):
    """
    Standardized demonstration batch-eval step. Returns all expected metrics and overlayable predictions.
    """
    network.eval()
    x = batch_data["image"].to(device).float()
    y = batch_data["label"].to(device).float()
    with torch.no_grad():
        outputs = network(x)
        loss = loss_fn(outputs, y).item()
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        y_true = y

        tp = (preds * y_true).sum().item()
        tn = ((1 - preds) * (1 - y_true)).sum().item()
        fp = (preds * (1 - y_true)).sum().item()
        fn = ((1 - preds) * y_true).sum().item()

        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
        precision = tp / (tp + fp + 1e-7) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn + 1e-7) if (tp + fn) > 0 else 0
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-7) if (2 * tp + fp + fn) > 0 else None
        per_class_dice = [dice]
        # Visualization: [input, prediction, label]
        input_np = x[0].cpu().numpy()
        label_np = y[0].cpu().numpy()
        pred_np = preds[0].cpu().numpy()
        viz = [{
            "input": input_np,
            "prediction": pred_np,
            "label": label_np
        }]
    return (dice, loss, per_class_dice, None, accuracy, precision, recall, viz)

def save_trained_model(network, params, model_dir="trained_models", model_name=None):
    """
    Save the trained model’s state dict for reproducible demo and future bundle export.
    """
    os.makedirs(model_dir, exist_ok=True)
    if model_name is None:
        model_id = params.get("model_id", "pathology_tumor_detection")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_id}_{timestamp}_final.pt"
    model_path = os.path.join(model_dir, model_name)
    torch.save(network.state_dict(), model_path)
    return model_path
