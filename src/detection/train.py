import torch
import yaml
import albumentations as A
import wandb

from pathlib import Path
from torchmetrics.detection import MeanAveragePrecision
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.data_processing.dataset import Manga109Dataset
from src.detection.detection_model import faster_rcnn
from src.detection.utils import save_checkpoint, load_checkpoint, clean_ram

device = "cuda" if torch.cuda.is_available() else "cpu"
def collate_fn(batch):
    return tuple(zip(*batch))

# Config 
yaml_path = "./configs/faster_rcnn_default.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Dataset
train_transform = A.Compose([
    A.Blur(blur_limit=3, p=0.2),
    A.Affine(rotate=(-5, 5), scale=(0.6, 1.4), p=0.4),
    A.RandomSizedBBoxSafeCrop(width=1024, height=1024, erosion_rate=0, p=0.2),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.CoarseDropout(max_holes=8, max_height=64, max_width=64, fill_value=255, p=0.2),
    ToTensorV2(),
], bbox_params=A.BboxParams(
    format="pascal_voc",
    label_fields=["labels"],
    min_visibility=0.3,
    min_area=100,
))
val_transform = A.Compose([
    ToTensorV2()
], bbox_params=A.BboxParams(
    format="pascal_voc",
    label_fields=["labels"],
))

train_dataset = Manga109Dataset(
    json_file=config["data"]["train_json"],
    img_dir=config["data"]["img_dir"],
    transforms=train_transform
)
val_dataset = Manga109Dataset(
    json_file=config["data"]["val_json"],
    img_dir=config["data"]["img_dir"],
    transforms=val_transform
)

# DataLoader
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    num_workers=config["data"]["num_workers"],
    collate_fn=collate_fn
)
val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=False,
    num_workers=config["data"]["num_workers"],
    collate_fn=collate_fn
)

def train_step(train_dataloader, model, device, optimizer):
    model.train()

    epoch_loss = 0.0
    loop = tqdm(train_dataloader, desc="Training", colour="cyan", total=len(train_dataloader))

    for i, (imgs, targets) in enumerate(loop):
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

        loss_dict = model(imgs, targets)
        if len(loss_dict) == 0:
            continue
        
        losses = sum(loss for loss in loss_dict.values())
        epoch_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        del imgs, targets, loss_dict
        if i % 100 == 0:
            clean_ram()

    epoch_loss /= len(train_dataloader)
    return epoch_loss

def val_step(val_dataloader, model, device):
    model.eval()
    mAP = MeanAveragePrecision()
    loop = tqdm(val_dataloader, desc="Evaluating...", colour="pink", total=len(val_dataloader))

    with torch.inference_mode():
        for imgs, targets in loop:
            imgs = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            outputs = model(imgs)
            targets = [{k: v.to("cpu") for k, v in target.items()} for target in targets]
            outputs = [{k: v.to("cpu") for k, v in output.items()} for output in outputs]

            mAP.update(outputs, targets)

            del imgs, targets, outputs
        
    results = mAP.compute()
    result = results["map"].item()
    del mAP
    
    return result

def train(train_dataloader, val_dataloader, model, optimizer, scheduler, device, epochs, model_save_path, checkpoint_save_path, use_wandb=True):
    results = {
        "train_loss": [],
        "test_metric": []
    }
    start_epoch = 0
    curr_best_metric = 0
    epochs = config["training"]["num_epochs"]

    if use_wandb:
        wandb.init(
            project="manga_translator_detection",
            name="",
            # id="8x4t37au",
            # resume="must",
            config={}
        )
    
    if checkpoint_save_path.exists():
        print(f"Resume training from {checkpoint_save_path}...")
        results, start_epoch, curr_best_metric = load_checkpoint(model, optimizer, scheduler, device, checkpoint_save_path)
    
    for epoch in tqdm(range(start_epoch, epochs), colour="orange", desc="Training and evaluating"):
        torch.cuda.ipc_collect()
        print(f"Epoch {epoch + 1}")

        train_loss = train_step(train_dataloader, model, device, optimizer)
        
        clean_ram()
        
        test_metric = val_step(val_dataloader, model, device)
        print(f"Train loss: {train_loss:.5f}, Test mAP50-95: {round(test_metric * 100, 4)}%")

        if use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "val_mAP_5095": test_metric, 
                "epoch": epoch + 1
            })
        
        results["train_loss"].append(train_loss)
        results["test_metric"].append(test_metric)

        if test_metric > curr_best_metric:
            curr_best_metric = test_metric
            torch.save(obj=model.state_dict(), f=model_save_path)
            print(f"Better than the current model, saving new model to {model_save_path}")

            if use_wandb:
                wandb.save(str(model_save_path), base_path="/kaggle/working", policy='now') 
                print("Uploaded Best Model to WandB Cloud!")

        scheduler.step()
        save_checkpoint(model, optimizer, scheduler, results, epoch + 1, curr_best_metric, checkpoint_save_path)
        if use_wandb:
            wandb.save(str(checkpoint_save_path), base_path="/kaggle/working", policy='live')
        
        del test_metric, train_loss
        clean_ram()

    return results


# Initialization
model = faster_rcnn(
    num_classes=config["model"]["num_classes"],
    anchor_sizes=config["model"]["anchor_sizes"],
    anchor_ratios=config["model"]["anchor_ratios"],
    box_nms_thresh=config["model"]["box_nms_thresh"]
)

optimizer = torch.optim.SGD(
    params=model.parameters(),
    lr=config["training"]["learning_rate"],
    momentum=config["training"]["momentum"],
    weight_decay=config["training"]["weight_decay"],
)

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer=optimizer,
    milestones=[20, 40],
    gamma=0.1
)
model_save_path = Path("")
checkpoint_save_path = Path("")
epochs = config["training"]["num_epochs"]

train(train_dataloader, val_dataloader, model, optimizer, scheduler, device, epochs, model_save_path, checkpoint_save_path, use_wandb=True)