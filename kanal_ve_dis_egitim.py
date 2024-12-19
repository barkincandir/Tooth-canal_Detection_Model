import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. Veri Seti Sınıfı
class CombinedDataset(Dataset):
    def __init__(self, kanal_images_dir, kanal_annotations_dir, dis_images_dir, dis_annotations_dir, transforms=None):
        self.kanal_images_dir = kanal_images_dir
        self.kanal_annotations_dir = kanal_annotations_dir
        self.dis_images_dir = dis_images_dir
        self.dis_annotations_dir = dis_annotations_dir
        self.transforms = transforms

        # İki veri setinin dosyalarını birleştir
        self.image_files = (sorted(os.listdir(kanal_images_dir)) +
                            sorted(os.listdir(dis_images_dir)))
        self.annotation_files = (sorted(os.listdir(kanal_annotations_dir)) +
                                 sorted(os.listdir(dis_annotations_dir)))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if idx < len(os.listdir(self.kanal_images_dir)):
            # Kanal verisi
            img_path = os.path.join(self.kanal_images_dir, self.image_files[idx])
            annotation_path = os.path.join(self.kanal_annotations_dir, self.annotation_files[idx])
            label = 1  # Kanal sınıfı
        else:
            # Diş verisi
            img_idx = idx - len(os.listdir(self.kanal_images_dir))
            img_path = os.path.join(self.dis_images_dir, self.image_files[img_idx])
            annotation_path = os.path.join(self.dis_annotations_dir, self.annotation_files[img_idx])
            label = 2  # Diş sınıfı

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = F.to_tensor(img)

        with open(annotation_path) as f:
            annotations = json.load(f)

        height, width = img.shape[1:]
        masks = []
        boxes = []
        for shape in annotations["shapes"]:
            if shape["shape_type"] == "line" or shape["shape_type"] == "polygon":
                points = np.array(shape["points"])
                mask = np.zeros((height, width), dtype=np.uint8)
                if shape["shape_type"] == "line":
                    for i in range(len(points) - 1):
                        pt1 = tuple(map(int, points[i]))
                        pt2 = tuple(map(int, points[i + 1]))
                        cv2.line(mask, pt1, pt2, color=1, thickness=3)
                elif shape["shape_type"] == "polygon":
                    cv2.fillPoly(mask, [points.astype(np.int32)], color=1)
                masks.append(mask)

                x_min, y_min = points.min(axis=0)
                x_max, y_max = points.max(axis=0)

                if x_max > x_min and y_max > y_min:
                    boxes.append([x_min, y_min, x_max, y_max])

        if masks:
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        else:
            masks = torch.zeros((0, height, width), dtype=torch.uint8)

        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.ones((len(masks),), dtype=torch.int64) * label  # Kanal: 1, Diş: 2

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target


# 2. Model Tanımı
def get_model(num_classes):
    # ResNet-101 backbone ile Mask R-CNN oluşturulur
    backbone = resnet_fpn_backbone(backbone_name='resnet101', weights="IMAGENET1K_V1")
    model = MaskRCNN(backbone=backbone, num_classes=num_classes)
    return model


# 3. Eğitim Döngüsü
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    for imgs, targets in data_loader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
    return total_loss / len(data_loader)


# 4. Model Eğitimi
def train_model(kanal_images_dir, kanal_annotations_dir, dis_images_dir, dis_annotations_dir, model_path, num_epochs=10):
    dataset = CombinedDataset(kanal_images_dir, kanal_annotations_dir, dis_images_dir, dis_annotations_dir)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=3)  # 1: Kanal, 2: Diş, 0: Arka Plan
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, optimizer, data_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    torch.save(model.state_dict(), model_path)


# 5. Çalıştırma
if __name__ == "__main__":
    kanal_images_dir = "C:\\Users\\candi\\Desktop\\27\\deneme1\\kanalgoruntu"  # Kanal görüntüleri
    kanal_annotations_dir = "C:\\Users\\candi\\Desktop\\27\\deneme1\\kanal_etiketleri"  # Kanal etiketleri
    dis_images_dir = "C:\\Users\\candi\\Desktop\\27\\deneme1\\disgoruntu"  # Diş görüntüleri
    dis_annotations_dir = "C:\\Users\\candi\\Desktop\\27\\deneme1\\dis_etiketleri"  # Diş etiketleri
    model_path = "kanal_ve_dis_model1.pth"

    train_model(kanal_images_dir, kanal_annotations_dir, dis_images_dir, dis_annotations_dir, model_path)
