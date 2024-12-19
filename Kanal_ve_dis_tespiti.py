import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import json


# 1. Model Yükleme
def load_model(model_path, num_classes=3):
    backbone = resnet_fpn_backbone(backbone_name='resnet101', weights="IMAGENET1K_V1")
    model = MaskRCNN(backbone=backbone, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# IoU Hesaplama Fonksiyonu
def calculate_iou(pred_box, gt_box):
    x1 = max(pred_box[0], gt_box[0])
    y1 = max(pred_box[1], gt_box[1])
    x2 = min(pred_box[2], gt_box[2])
    y2 = min(pred_box[3], gt_box[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union = pred_area + gt_area - intersection

    return intersection / union if union != 0 else 0


# Maskeden Bounding Box'a Dönüşüm
def mask_to_bbox(mask):
    y_coords, x_coords = np.where(mask)
    if len(y_coords) == 0 or len(x_coords) == 0:
        return None
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    return [x_min, y_min, x_max, y_max]


# JSON'dan Gerçek Etiketleri Yükleme Fonksiyonu
def load_ground_truth_from_json(json_path, label_type):
    ground_truth_boxes = []
    with open(json_path, 'r') as file:
        data = json.load(file)
        for shape in data["shapes"]:
            if shape["label"] == label_type:
                points = np.array(shape["points"])
                x_min, y_min = points.min(axis=0)
                x_max, y_max = points.max(axis=0)
                ground_truth_boxes.append([x_min, y_min, x_max, y_max])
    return ground_truth_boxes
def load_lines_from_json(json_path, label_type):
    lines = []
    with open(json_path, 'r') as file:
        data = json.load(file)
        for shape in data["shapes"]:
            if shape["label"] == label_type and shape["shape_type"] == "line":
                lines.append(shape["points"])
    return lines
def load_polygons_from_json(json_path, label_type):
    polygons = []
    with open(json_path, 'r') as file:
        data = json.load(file)
        for shape in data["shapes"]:
            if shape["label"] == label_type and shape["shape_type"] == "polygon":
                polygons.append(shape["points"])
    return polygons
def calculate_box_iou(pred_box, gt_box):
    x1 = max(pred_box[0], gt_box[0])
    y1 = max(pred_box[1], gt_box[1])
    x2 = min(pred_box[2], gt_box[2])
    y2 = min(pred_box[3], gt_box[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union = pred_area + gt_area - intersection

    return intersection / union if union > 0 else 0
def polygon_to_bbox(polygon):
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return [x_min, y_min, x_max, y_max]


def calculate_line_iou(pred_line, gt_line):
    """
    İki çizgi arasındaki IoU benzeri bir ölçüm hesaplar.
    """
    # Her iki çizginin başlangıç ve bitiş noktalarını alın
    pred_start, pred_end = np.array(pred_line[0]), np.array(pred_line[1])
    gt_start, gt_end = np.array(gt_line[0]), np.array(gt_line[1])

    # Çizgilerin dikey eksendeki kapsamını hesapla
    pred_y_min, pred_y_max = min(pred_start[1], pred_end[1]), max(pred_start[1], pred_end[1])
    gt_y_min, gt_y_max = min(gt_start[1], gt_end[1]), max(gt_start[1], gt_end[1])

    # Overlap mesafesini hesapla (iki çizginin dikey eksende çakıştığı mesafe)
    overlap = max(0, min(pred_y_max, gt_y_max) - max(pred_y_min, gt_y_min))

    # Çizgilerin uzunluklarını hesapla
    pred_length = pred_y_max - pred_y_min
    gt_length = gt_y_max - gt_y_min

    # IoU Hesabı
    union = pred_length + gt_length - overlap
    return overlap / union if union > 0 else 0

def mask_to_line(mask):
    """
    Maskeden çizgiye dönüştürme. Çizgi, maskenin üst ve alt noktalarının ortalama x koordinatına göre hesaplanır.
    """
    y_coords, x_coords = np.where(mask)
    if len(y_coords) == 0 or len(x_coords) == 0:
        return None
    top_point = (np.mean(x_coords[y_coords == y_coords.min()]), y_coords.min())
    bottom_point = (np.mean(x_coords[y_coords == y_coords.max()]), y_coords.max())
    return [top_point, bottom_point]

# Etkileşimli Görselleştirme
class InteractiveVisualizer:
    def __init__(self, image, kanal_masks, dis_boxes):
        self.image = image
        self.original_image = image.copy()
        self.kanal_masks = kanal_masks
        self.dis_boxes = dis_boxes
        self.show_kanal = True
        self.show_dis = True

    def draw(self):
        img = self.original_image.copy()

        # Kanal maskelerini ve numaralarını çiz
        if self.show_kanal:
            for i, mask in enumerate(self.kanal_masks):
                mask = mask[0] > 0.2  # Binarize maske
                y_coords, x_coords = np.where(mask)

                if len(y_coords) == 0 or len(x_coords) == 0:
                    continue

                top_point = (int(np.mean(x_coords[y_coords == y_coords.min()])), y_coords.min())
                bottom_point = (int(np.mean(x_coords[y_coords == y_coords.max()])), y_coords.max())
                cv2.line(img, top_point, bottom_point, (0, 0, 255), 2)  # Kanal için çizgi

                kanal_text = f"K {i + 1}"
                cv2.putText(img, kanal_text, (top_point[0], top_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Diş kutucuklarını ve numaralarını çiz
        if self.show_dis:
            for i, box in enumerate(self.dis_boxes):
                x_min, y_min, x_max, y_max = map(int, box)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                dis_text = f"D {i + 1}"
                cv2.putText(img, dis_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Sağ alt köşede toplam kanal ve diş sayısını yazdır
        kanal_count = len(self.kanal_masks)
        dis_count = len(self.dis_boxes)
        count_text = f"Tespit Edilen: Kanallar: {kanal_count}, Dişler: {dis_count}"
        self.ax.text(0.98, 0.02, count_text, color="white", fontsize=10,
                     verticalalignment="bottom", horizontalalignment="right",
                     bbox=dict(facecolor="black", alpha=0.7, edgecolor="none"), transform=self.ax.transAxes)

        self.ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.ax.axis("off")
        plt.draw()

    def toggle_kanal(self, event):
        self.show_kanal = not self.show_kanal
        self.draw()

    def toggle_dis(self, event):
        self.show_dis = not self.show_dis
        self.draw()

    def visualize(self):
        fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)

        self.draw()

        ax_kanal = plt.axes([0.1, 0.05, 0.2, 0.075])
        btn_kanal = Button(ax_kanal, 'Kanal Göster/Kapat')
        btn_kanal.on_clicked(self.toggle_kanal)

        ax_dis = plt.axes([0.35, 0.05, 0.2, 0.075])
        btn_dis = Button(ax_dis, 'Diş Göster/Kapat')
        btn_dis.on_clicked(self.toggle_dis)

        plt.show()


# Tahmin ve Görselleştirme
def predict_and_visualize(image_path, kanal_json_path, dis_json_path, model, device, kanal_threshold=0.3, dis_threshold=0.5):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    masks = outputs[0]["masks"].cpu().numpy()
    scores = outputs[0]["scores"].cpu().numpy()
    labels = outputs[0]["labels"].cpu().numpy()
    boxes = outputs[0]["boxes"].cpu().numpy()

    kanal_indices = (labels == 1) & (scores > kanal_threshold)
    dis_indices = (labels == 2) & (scores > dis_threshold)

    kanal_masks = masks[kanal_indices]
    kanal_scores = scores[kanal_indices]
    dis_boxes = boxes[dis_indices]
    dis_scores = scores[dis_indices]

    # Kanallar için bounding box oluştur
    kanal_boxes = [mask_to_bbox(mask[0]) for mask in kanal_masks]

    # JSON'dan gerçek kutuları yükle
    kanal_ground_truth_boxes = load_ground_truth_from_json(kanal_json_path, "k1")
    dis_ground_truth_boxes = load_ground_truth_from_json(dis_json_path, "d1")

    # Kanal bilgilerini yazdır
    print("\nKanallar:")
    if len(kanal_scores) > 0:
        print(f"Max Doğruluk: %{kanal_scores.max() * 100:.2f}")
        print(f"Min Doğruluk: %{kanal_scores.min() * 100:.2f}")
        for i, (box, score) in enumerate(zip(kanal_boxes, kanal_scores)):
            if box is not None:
                print(f"Kanal {i + 1}: [x1: {box[0]:.2f}, y1: {box[1]:.2f}, x2: {box[2]:.2f}, y2: {box[3]:.2f}] Score: %{score * 100:.2f}")
    else:
        print("Kanallar bulunamadı.")

    # Diş bilgilerini yazdır
    print("\nDişler:")
    if len(dis_scores) > 0:
        print(f"Max Doğruluk: %{dis_scores.max() * 100:.2f}")
        print(f"Min Doğruluk: %{dis_scores.min() * 100:.2f}")
        for i, (box, score) in enumerate(zip(dis_boxes, dis_scores)):
            x_min, y_min, x_max, y_max = box
            print(f"Diş {i + 1}: [x1: {x_min:.2f}, y1: {y_min:.2f}, x2: {x_max:.2f}, y2: {y_max:.2f}] Score: %{score * 100:.2f}")
    else:
        print("Dişler bulunamadı.")

    # Kanal IoU hesaplama ve yazdırma
    print("\n")
    kanal_iou_scores = []
    predicted_lines = [mask_to_line(mask[0]) for mask in kanal_masks if mask_to_line(mask[0]) is not None]
    ground_truth_lines = load_lines_from_json(kanal_json_path, "k1")  # JSON'dan yükleme

    for i, pred_line in enumerate(predicted_lines):
        best_iou = 0
        for gt_line in ground_truth_lines:
            iou = calculate_line_iou(pred_line, gt_line)
            best_iou = max(best_iou, iou)
        kanal_iou_scores.append(best_iou)
    if kanal_iou_scores:
        print(f"Ortalama Kanal IoU: {np.mean(kanal_iou_scores):.2f}")

    # Diş IoU hesaplama ve yazdırma
    print("\n")
    dis_iou_scores = []
    predicted_boxes = dis_boxes  # Modelden gelen tahminler (Box formatında)
    ground_truth_polygons = load_polygons_from_json(dis_json_path, "d1")  # JSON'dan yükleme
    ground_truth_boxes = [polygon_to_bbox(poly) for poly in ground_truth_polygons]

    for i, pred_box in enumerate(predicted_boxes):
        best_iou = 0
        for gt_box in ground_truth_boxes:
            iou = calculate_box_iou(pred_box, gt_box)
            best_iou = max(best_iou, iou)
        dis_iou_scores.append(best_iou)
    print(f"Ortalama Diş IoU: {np.mean(dis_iou_scores):.2f}")

    # Görselleştirme
    visualizer = InteractiveVisualizer(img, kanal_masks, dis_boxes)
    visualizer.visualize()



def main():
    model_path = "kanal_ve_dis_model.pth"
    image_path = "C:\\Users\\candi\\Desktop\\27\\deneme1\\disgoruntu\\dis (1).JPG"
    kanal_json_path = "C:\\Users\\candi\\Desktop\\27\\deneme1\\kanal_etiketleri\\kanal (1).json"
    dis_json_path = "C:\\Users\\candi\\Desktop\\27\\deneme1\\dis_etiketleri\\dis (1).json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, num_classes=3)
    model.to(device)

    predict_and_visualize(image_path, kanal_json_path, dis_json_path, model, device, kanal_threshold=0.3, dis_threshold=0.5)


if __name__ == "__main__":
    main()
