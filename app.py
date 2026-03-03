import gradio as gr
import torch
import yaml
import numpy as np

from PIL import Image
from src.detection.detection_model import faster_rcnn
from src.ocr.ocr_system import MangaTextExtractor
from src.translation.translator_system import MangaTranslator
from src.translation.renderer_system import MangaRenderer

CONFIG_PATH = "./configs/faster_rcnn_default.yaml"
DETECTION_WEIGHTS_PATH = "./models/faster_rcnn_default_weights.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

detection_model = faster_rcnn(
    num_classes=config["model"]["num_classes"],
    anchor_sizes=config["model"]["anchor_sizes"],
    anchor_ratios=config["model"]["anchor_ratios"],
    box_nms_thresh=config["model"]["box_nms_thresh"] 
)

detection_model.load_state_dict(torch.load(DETECTION_WEIGHTS_PATH, map_location=device))
detection_model.to(device)
detection_model.eval()

ocr_model = MangaTextExtractor()
translator_module = MangaTranslator("ja", "en")
renderer_module = MangaRenderer()

def apply_containment_filter(boxes, labels, threshold=config["model"]["confidence_thresh"]):
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

def translate_manga_page(input_img_path):
    output_img_path = "./outputs/temp_output.jpg"
    confidence_threshold = 0.6

    img = Image.open(input_img_path).convert("RGB")
    img_np = np.array(img)
    img_tensor = torch.tensor(img_np / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.inference_mode():
        predictions = detection_model(img_tensor)[0]

    keep_idx = predictions["scores"] >= confidence_threshold
    boxes = predictions["boxes"][keep_idx].cpu().numpy()
    labels = predictions["labels"][keep_idx].cpu().numpy()

    boxes, labels = apply_containment_filter(boxes, labels, threshold=0.8)

    text_boxes = [box for box, label in zip(boxes, labels) if label == 2]
    frame_boxes = [box for box, label in zip(boxes, labels) if label == 1]

    # OCR
    ocr_results = ocr_model.extract_text(input_img_path, text_boxes, frame_boxes)

    # Translation
    translated_data = translator_module.translate_with_context(ocr_results)

    # Render
    renderer_module.render_translated_image(input_img_path, translated_data, output_img_path)

    return output_img_path

with gr.Blocks() as demo:
    gr.Markdown()
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="filepath", label="Input Your Manga Page HERE!")
            translate_btn = gr.Button("🚀 TRANSLATE!", variant="primary")
            
        with gr.Column():
            output_image = gr.Image(type="filepath", label="Translated Manga Page")

    translate_btn.click(
        fn=translate_manga_page,
        inputs=input_image,
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), share=False) 