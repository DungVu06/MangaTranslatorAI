import torch
import yaml
import numpy as np
import argparse

from pathlib import Path
from PIL import Image
from src.detection.detection_model import faster_rcnn
from src.ocr.ocr_system import MangaTextExtractor
from src.translation.translator_system import MangaTranslator
from src.translation.renderer_system import MangaRenderer

INPUT_DIR = Path("./data/input")
OUTPUT_DIR = Path("./outputs")
CONFIG_PATH = "./configs/faster_rcnn_default.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_trained_model(config, device):
    model = faster_rcnn(
        num_classes=config["model"]["num_classes"],
        anchor_sizes=config["model"]["anchor_sizes"],
        anchor_ratios=config["model"]["anchor_ratios"],
        box_nms_thresh=config["model"]["box_nms_thresh"] 
    )

    model.load_state_dict(torch.load(config["model"]["weights_path"], map_location=device, weights_only=True))
    model.eval()

    return model

def apply_containment_filter(boxes, labels, threshold=0.8):
    n = len(boxes)
    if n == 0:
        return boxes, labels
        
    suppressed = np.zeros(n, dtype=bool)

    for i in range(n):
        if suppressed[i]:
            continue
        for j in range(i + 1, n):
            if suppressed[j]:
                continue

            if labels[i] != labels[j]:
                continue

            boxA, boxB = boxes[i], boxes[j]

            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            interArea = max(0, xB - xA) * max(0, yB - yA)

            if interArea > 0:
                areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
                min_area = min(areaA, areaB)

                if interArea / min_area > threshold:
                    if areaA >= areaB:
                        suppressed[j] = True
                    else:
                        suppressed[i] = True
                        break

    keep_indices = [i for i, s in enumerate(suppressed) if not s]
    
    return boxes[keep_indices], labels[keep_indices]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Input image file name")
    args = parser.parse_args()

    input_path = INPUT_DIR / args.filename
    output_path = OUTPUT_DIR / f"{input_path.stem}_translated{input_path.suffix}"
    INPUT_IMG_PATH = str(input_path)
    OUTPUT_IMG_PATH = str(output_path)

    detection_model = load_trained_model(config, device)
    ocr_model = MangaTextExtractor()
    translator_module = MangaTranslator("ja", "en")
    renderer_module = MangaRenderer()

    # Detection
    img = Image.open(INPUT_IMG_PATH).convert("RGB")
    img_np = np.array(img)
    img_tensor = torch.tensor(img_np / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    with torch.inference_mode():
        predictions = detection_model(img_tensor)[0]

    keep_idx = predictions["scores"] >= config["model"]["confidence_thresh"]
    boxes = predictions["boxes"][keep_idx].cpu().numpy()
    labels = predictions["labels"][keep_idx].cpu().numpy()
    boxes, labels = apply_containment_filter(boxes, labels, threshold=0.8)

    text_boxes = []
    frame_boxes = []
    for box, label in zip(boxes, labels):
        if label == 1:
            frame_boxes.append(box)
        else:
            text_boxes.append(box)

    # OCR 
    ocr_results = ocr_model.extract_text(INPUT_IMG_PATH, text_boxes, frame_boxes)

    # Translation
    translated_data = translator_module.translate_with_context(ocr_results)

    # Render
    renderer_module.render_translated_image(INPUT_IMG_PATH, translated_data, OUTPUT_IMG_PATH)