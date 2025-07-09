# 🔒 Privacy Vision Pipeline

This project presents a **privacy-aware image redaction pipeline** that intelligently detects and obfuscates sensitive visual content such as faces, license plates, addresses, screens, and legible text using a combination of **YOLOv11-seg**, **OCR**, and **LLMs (LLaMA3)**.

We design a modular pipeline that quantifies the **privacy risk** of each region using both traditional image metrics and semantic cues via LLM inference, applying Gaussian blur or inpainting based on thresholds.

This is the Research Report: [Automatic Image Processing for Privacy](https://github.com/Aarya-Kul/Automatic-privacy-vision/blob/main/Image_Privacy_Research_Paper.pdf)

---

## 💡 Motivation

With the surge in publicly available visual content (e.g., street photos, social media, surveillance), it is essential to protect sensitive information within images. Traditional object detectors often fail in ambiguous or low-resolution cases, and lack contextual awareness.

This project introduces a hybrid **CV + NLP** pipeline that scores the privacy sensitivity of detected regions by combining:
- Visual heuristics (e.g., blurriness, center bias, region size)
- OCR confidence and legibility
- Semantic analysis from a **local LLM (LLaMA3)**

---

## 📦 What’s Inside

- ✅ **YOLOv11-seg** for object segmentation of privacy-sensitive classes
- 🔍 **EasyOCR** for extracting readable text
- 🧠 **LLM Reasoning** using local LLaMA3 (via Ollama)
- 📊 Weighted region scoring and class-specific rules
- 🧼 Redaction via **Gaussian blur** or **inpainting**
- 📁 CLI-ready and batch image processing support

---

## 🧱 Pipeline Architecture

```
[ Image ]
   ↓
[ YOLOv11-seg ] + [ EasyOCR ]
   ↓
[ Merge Boxes by IoU ]
   ↓
[ Region Scoring ]
   ↓
[ Redaction (blur or inpaint) ]
   ↓
[ Output ]
```

---

## 🧪 Region Scoring Features

| Feature           | Description                                  |
|------------------|----------------------------------------------|
| OCR Confidence    | How confidently text is recognized           |
| Blurriness        | Laplacian-based image sharpness              |
| Center Focus      | If object is near the center of the image    |
| Size Ratio        | Relative area of region in image             |
| Background Complexity | Color + texture entropy in region       |
| LLM Sensitivity   | LLaMA3 estimates sensitivity of detected text|

---

## 📂 Object Classes

| Class Name           | ID |
|----------------------|----|
| address              | 0  |
| business_sign        | 1  |
| electronicscreens    | 2  |
| face                 | 3  |
| license_plate        | 4  |
| personal_document    | 5  |
| photo                | 6  |
| street_name          | 7  |

> *Note*: The mapping previously included additional classes like `advertisement` and `legible_text`.

---

## 📁 Dataset Sources

- 🧾 **DIPA dataset** (core privacy detection)
- 🏙️ **Laion-400m** (for general scene images)
- 🏠 **Stanford Street View House Numbers (SVHN)** for ~400 house address images
- 🖍️ Annotations generated using [Roboflow](https://app.roboflow.com)

To download DIPA data:
```bash
bash data/dipa/setup.sh
```

---

## 🛠️ Setup Instructions

### 🔧 Python Dependencies

```bash
pip install -r requirements.txt
```

Also ensure:
- YOLO weights are saved at: `best_yolo_weights/best.pt`
- OCR model is properly installed (EasyOCR)

---

### 🧠 LLM Setup with Ollama

#### macOS
```bash
brew install ollama
```

#### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### Fix PATH (if needed)
```bash
echo >> ~/.zprofile
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

#### Start LLaMA3
```bash
ollama serve
ollama run llama3
```

Stop with:
```bash
sudo systemctl stop ollama
```

---

## ▶️ Running the Pipeline

### Default Usage

```bash
python pipeline.py
```

Place input images in `./pipeline_in`. Output will be saved in `./pipeline_out`.

### Custom Paths

```bash
python pipeline.py debug_pipeline -i -t debug_out
```

- `debug_pipeline/` → input folder
- `debug_out/` → output folder
- `-i` → use **inpainting** instead of blur (except for faces)
- `-p 0.3 0.5` → blur if score ≥ 0.3, inpaint if score ≥ 0.5

---

### CLI Options Summary

| Flag               | Description                                                   |
|--------------------|---------------------------------------------------------------|
| `-t <path>`         | Output directory (default: `pipeline_out`)                   |
| `-p <t1> [t2]`      | Privacy thresholds: blur ≥ `t1`, inpaint ≥ `t2`              |
| `-i`                | Use inpainting generally (blur still used for faces)         |
| `--dry`             | Run without saving redacted images (for scoring only)        |

Predictions from YOLO are saved to `<out>/yolo_predictions`.

> *Note*: Subdirectories are not scanned recursively.

---

## 🧾 Sample Inference

```bash
python pipeline.py samples/ -i -p 0.3 0.5 -t output/
```

---

## 📌 Future Work

- Train feature fusion models instead of hand-tuned weights
- Apply to video + temporal consistency tracking
- Incorporate larger open-source vision-language models (e.g. MiniGPT-4, LLaVA)
- Extend to support multilingual OCR & LLMs

---

## 👥 Authors

- Aarya Kulshrestha and Others
