# 🌸 Manga Translator AI 🌸

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)
![Manga-OCR](https://img.shields.io/badge/OCR-manga--ocr-ff69b4.svg)

An End-to-End AI pipeline that automatically detects, extracts, translates, and renders manga pages from Japanese to English.

## ✨ Pipeline Overview
1. **Object Detection:** Utilizes `Faster R-CNN` to detect Manga Frames and Speech Bubbles (Text Boxes).
2. **Reading Order Sorting:** A custom algorithm that automatically sorts the reading flow based on frame coordinates (Right-to-Left, Top-to-Bottom).
3. **OCR:** Integrates `manga-ocr` for highly accurate Japanese text extraction.
4. **Machine Translation:** Context-aware translation from Japanese to English using `deep-translator`.
5. **Rendering:** Automatically erases original text, calculates word-wrap, and seamlessly renders the translated English text back onto the original image.

---
## 👁️ Demo
  ![Translation demo](./demo/demo.gif)

## 🖼️ Results
| Original | Detection | Translated |
| :---: | :---: | :---: |
| ![Original](./demo/original_img.jpg) | ![Detection](./demo/detection_img.png) | ![Result](./demo/translated_img.jpg) |

## 📂 Project Structure

```text
.
└── MangaTranslator/
    ├── configs/ 
    │   └── faster_rcnn_default.yaml     # Configurations for data, model, and training
    ├── data/                            <-- [Git Ignored] Place your raw images and dataset here
    │   ├── raw
    │   ├── processed
    │   └── inference_data
    ├── models/                          <-- [Git Ignored] Place your downloaded .pt weights here
    │   └── faster_rcnn_default_weights.pt # Faster R-CNN weights (Download from Kaggle/Drive)
    ├── notebooks/
    │   └── test_dataset.ipynb           # Jupyter notebook for dataset exploration
    ├── outputs/                         # Directory for successfully translated images
    ├── src/
    │   ├── data_processing/
    │   │   ├── cleaning_file_name.py    # Fixes corrupted file names
    │   │   ├── convert_xml_to_coco.py   # Converts XML annotations to COCO format
    │   │   ├── dataset.py               # Custom PyTorch Dataset class
    │   │   └── split_dataset.py         # Splits data into train/val/test sets
    │   ├── detection/
    │   │   ├── detection_model.py       # Faster R-CNN model initialization
    │   │   ├── test.py                  # Evaluation phase on the test dataset
    │   │   ├── train.py                 # Training and validation phase
    │   │   └── utils.py                 # Helper functions
    │   ├── ocr/
    │   │   └── ocr_system.py            # manga-ocr model initialization and text extraction
    │   └── translation/
    │       ├── renderer_system.py       # Erases old text, applies word-wrap, and draws new text
    │       └── translator_system.py     # Translation module using deep_translator
    ├── .gitignore                       # Ignores __pycache__, heavy weights, data, and outputs
    ├── inference.py                     # Standalone script for testing Detection + OCR
    ├── main.py                          # Main entry point for the End-to-End pipeline
    └── requirements.txt                 # Libraries used
 
```

## 📊 Dataset
[Manga109s](https://huggingface.co/datasets/hal-utokyo/Manga109-s/tree/main)

## ⚙️ Installation
1. **Clone this repository:** 
```bash
git clone https://github.com/DungVu06/MangaTranslator.git
cd MangaTranslator
```
2. **Install the required dependencies:** 
``` bash
pip install -r requirements.txt
```
3. **Important Note on Detection Model Weights:** 
The `models/faster_rcnn_default_weights.pt` file is not included in this repository due to GitHub's file size limits. Please train the model yourself (using `src/detection/train.py`) or download the pre-trained weights and place them in the models/ directory before running the pipeline. Download link: [Google Drive](https://drive.google.com/file/d/1gzLn94gRnxJJjMBsZjFPCjoA76jD7XLp/view?usp=sharing)

## 🚀 Usage
1. **Quick Test**
To verify if the model correctly detects bounding boxes, reads Japanese text and translating:
``` bash
python inference.py FILE-NAME
```
2. **Full Pipeline (End-to-End Translation)**
To run the complete process from a raw Japanese manga page to a translated English page:
``` bash
python main.py FILE-NAME
```
* Input: the image file name in main.py (e.g., `.snk_1.jpg`). Notice: Input data must be stored in `./data/input`
* Output: The fully translated and rendered image will be saved in `./outputs/output.jpg`.

## 🧠 Current Challenges
  * **Contextual Constraints**: The model is currently limited to extracting text exclusively within speech bubbles.
  * **Geometric Sensitivity** (FIXED): Performance degrades significantly when processing non-rectangular or isolated single-frame inputs.
  * **Complex Scene Handling** (FIXED): The system struggles with high-density action scenes, overlapping characters, or cluttered text layouts.
  * **Text Reflow Issues** (FIXED): The text-wrapping logic is rudimentary, relying on fixed bounding boxes which often leads to unnatural line breaks. Text size is not flexible making it not optimized for different bubble's sizes.
  * **Rendering Artifacts**: The model struggles when a character occupies a significant portion of the speech bubble; the in-painting algorithm may inadvertently erase parts of the character's body (specifically following the text's bounding box, which results in an unnatural visual output). The text removal algorithm encounters significant difficulties when processing raw images (low-resolution or uncleaned source material).
  * **Performance Bottlenecks**: Overall execution speed is suboptimal due to the overhead of the Faster R-CNN architecture and external API latency. (~8s/img)
  * **Layout Sequencing** (FIXED): The sorting algorithm fails to correctly sequence text when processing double-page spreads (two pages per image) instead of individual pages.
  * **Font errors**: The model occasionally flags standalone punctuation (?, !, .) as Japanese text. This can cause the translation module to fail or trigger font rendering errors
  
## 🆙 Version History
* **v1.0.0:**
  * Successfully implemented a functional end-to-end pipeline; however, latency remains a significant bottleneck.
  * Core components, including Object Detection (Faster R-CNN), OCR, Translation, and Rendering, are operational and meet basic requirements.
* **v1.0.1:**
  * Updated demo in README.md using Gradio
* **v1.1.0:**
  * Better colors and cleaner text removal in the final translated images.
  * Minor updates to make running main.py and inference.py easier.
  * Update new font.
* **v1.2.0:**
  * Update new text wrapping/size algorithms for faster and more accurate translated text!
* **v1.3.0 (Current):**
  * A new Faster R-CNN model trained with aggressive data augmentation (harder transforms) to ensure robust performance across diverse manga art styles.
  * Implemented a new filtering logic in post-processing to prevent the model from detecting redundant small text boxes inside larger ones.
  * Significantly improved text detection accuracy within high-action frames.

## 🙏 Thanks for Watching
If you find this project interesting or useful, feel free to ⭐ star the repository and share your feedback.
Contributions and suggestions are always welcome!

I'm Yorha. Built with love for AI and Manga! 🎏🎎
