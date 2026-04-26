# Image Text Identifier

A local, privacy-first OCR app built with [EasyOCR](https://github.com/JaidedAI/EasyOCR) and [Streamlit](https://streamlit.io) — extract text from images entirely on your machine, no cloud, no API keys.

![Image Text Identifier demo](OCR.gif)

---

## Features

- **Upload or sample images** — drag-and-drop your own files or use the bundled sample set
- **Image preprocessing** — denoise, deskew, and sharpen before OCR to improve accuracy on noisy or skewed scans
- **Confidence-based annotation** — bounding boxes colour-coded green / orange / red by detection confidence
- **Adjustable confidence threshold** — filter low-confidence detections from the output at runtime
- **Detection details table** — per-region text, confidence percentage, and bounding box coordinates
- **Export results** — download extracted text as `.txt` or structured detections as `.json`
- **Trainable correction model** — label OCR output with ground-truth text, train a word-level correction model, and save it as a `.pkl` that auto-applies on future runs
- **Incremental training** — each "Train & Save" merges new corrections into the existing model without overwriting prior work
- **Model portability** — download and upload correction models (`.pkl`) directly from the sidebar

---

## Project Structure

```
Image_Text_Identifier/
├── app.py              # Streamlit UI — OCR tab, Train tab, sidebar controls
├── pipeline.py         # EasyOCR wrapper — run_ocr, draw_results, results_to_text
├── preprocess.py       # OpenCV preprocessing — denoise, deskew, sharpen
├── trainer.py          # OCRCorrectionModel — fit, predict, save/load pickle
├── requirements.txt
└── ocr/                # Bundled sample images
    ├── image1.png
    ├── image2.png
    └── image3.png
```

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Usage

### OCR Tab

1. Choose **Upload image** to drag-and-drop your own files, or **Use sample images** to pick from the bundled set.
2. Adjust the **confidence threshold** slider in the sidebar to filter detections.
3. Toggle **preprocessing** options (denoise / deskew / sharpen) to improve results on noisy or skewed scans.
4. View the annotated image and extracted text side-by-side.
5. Expand **Detection details** for a full per-region breakdown.
6. Download results as `.txt` or `.json`.

If a correction model is loaded (see Train tab), it is automatically applied to all OCR output before display.

### Train Correction Model Tab

Use this tab to teach the app about recurring OCR mistakes specific to your documents or fonts.

1. Upload one or more training images.
2. For each image, the app runs OCR and displays the raw output on the left.
3. Type the **correct text** in the ground-truth field on the right.
4. Click **Add to training data** — the app aligns OCR words with correct words using `difflib` and records differing pairs.
5. Repeat for as many images as needed; accumulated pairs are shown in the session counter.
6. Click **Train & Save Model** to fit the correction model and save it as `ocr_model.pkl` in the project root.

On the next OCR run the model is loaded automatically. Incremental training is supported — each save merges new corrections into the existing model.

---

## Sidebar Settings

| Setting | Default | Description |
|---|---|---|
| Confidence threshold | 0.30 | Detections below this score are hidden from output |
| Denoise | On | Applies fast non-local means denoising (OpenCV) |
| Deskew | Off | Detects and corrects page rotation angle |
| Sharpen | Off | Applies a sharpening convolution kernel |

Annotation colours: **green** ≥ 80 % · **orange** ≥ 50 % · **red** < 50 %

---

## Preprocessing Pipeline

| Step | Method | When to use |
|---|---|---|
| Denoise | `cv2.fastNlMeansDenoisingColored` | Scanned documents, noisy photos |
| Deskew | `cv2.minAreaRect` + `warpAffine` | Rotated pages or handheld camera shots |
| Sharpen | 3×3 Laplacian kernel | Blurry or low-resolution images |

---

## Correction Model

The correction model (`OCRCorrectionModel` in `trainer.py`) applies word-level corrections in order:

1. **Exact match** — if the OCR'd word exists in the corrections dictionary, it is replaced directly.
2. **Fuzzy match** — if not found exactly, `difflib.get_close_matches` searches the learned vocabulary (cutoff 0.75).
3. **Passthrough** — if no match is found, the word is returned unchanged.

The model is serialised with Python's `pickle` module and saved as `ocr_model.pkl`. You can download, share, or replace it from the sidebar.

---

## Export Formats

### `.txt`

Plain text, one detected line per newline, filtered by the current confidence threshold.

### `.json`

```json
[
  {
    "text": "Hello World",
    "confidence": 0.97,
    "bbox": [[10, 5], [120, 5], [120, 22], [10, 22]]
  }
]
```

`bbox` is a four-point polygon `[top-left, top-right, bottom-right, bottom-left]` in pixel coordinates.
