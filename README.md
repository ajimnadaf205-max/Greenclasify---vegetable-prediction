# greenclasify

Vegetable image classifier with a lightweight web UI.

This repository contains a PyTorch model (MobileNetV2-based) trained to classify 15 vegetables, a Flask backend that serves predictions, and a curated HTML/CSS/JS frontend (dashboard style) for users to upload images and view results and nutritional info.

dataset download link: https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset
---

## Quick Start

Requirements (tested on Windows):
- Python 3.10+ with virtualenv
- GPU optional (CUDA) but CPU works

Install dependencies and run the server:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python web_app.py
```

Open your browser at http://localhost:5000

Notes:
- `web_app.py` runs the Flask server (port 5000 by default).
- `app.py` is an earlier Streamlit UI entrypoint; you can keep it (alternative UI) or remove it if you prefer only the Flask + HTML UI.

---

## Project Structure (files and folders)

- `web_app.py` — Flask application and `/predict` API.
  - Loads the trained model via `get_model()` from `model.py` and expects `vegetable_classifier.pth` in project root.
  - `/` renders the HTML dashboard in `templates/index.html`.
  - `/predict` accepts an uploaded image file (form field `image`) and returns top-5 predictions plus nutritional info in JSON.

- `templates/index.html` — Dashboard front-end HTML.
  - Left sidebar with navigation: Home, Predict, About.
  - `Home` contains a brief description and sample photos.
  - `Predict` contains image upload, preview, top prediction, small confidence chart, and nutrition details.
  - `About` contains project info.

- `static/styles.css` — Styles for the dashboard (light theme).

- `static/script.js` — Client-side logic:
  - Page navigation between Home/Predict/About.
  - Handles image selection, preview, POST to `/predict`, and Chart.js rendering of top predictions.

- `model.py` — (existing) contains the model construction function `get_model(num_classes)`.
  - This file is imported by `web_app.py` to build the same architecture used during training.

- `train.py` — (existing) training script (if present) used for model training. See its docstring/comments for training-specific commands and hyperparameters.

- `data_loader.py` — (existing) data loading utilities that prepare images and datasets for training/validation/test.

- `confusion_matrix.py` — (existing) evaluation/visualization utilities for model performance analysis.

- `vegetable_classifier.pth` — Pretrained model weights (expected at project root). This file is required to run inference. If you want to retrain, use `train.py` and save new weights to this path (or update `web_app.py` to point to a different path).

- `Vegetable Images/` — Dataset folders organized by class (train/validation/test splits are provided under the workspace). These folders are used for training.

- `requirements.txt` — List of Python dependencies. Includes `Flask`, `torch`, `torchvision`, `opencv-python`, `Pillow`, etc.

- `app.py` — Streamlit-based UI (legacy). Keep if you want an alternate Streamlit interface.

---

## API specification

POST /predict
- Content-Type: multipart/form-data
- Field: `image` — binary image file (jpg, png, etc.)

Response (JSON):
```
{
  "predictions": [ {"class": "Tomato", "prob": 0.84}, ... ],
  "nutrition": { "Calories": "18 kcal", "Vitamins": "Vitamin C, K" }
}
```

Client-side `script.js` consumes this response to show the top prediction, a small chart (top-5), and nutrition data.

---

## Details and Implementation Notes

- Model input preprocessing: images are resized to 224x224, normalized with ImageNet mean/std (0.485,0.456,0.406 and 0.229,0.224,0.225). Ensure that incoming images are RGB.
- Prediction code uses `torch.softmax` on model outputs and returns top-k probabilities.
- The Flask server moves the model to the detected device (`cuda` if available) but runs inference on CPU if no GPU exists.

Performance/Tips:
- For faster cold-start, after first request, the model remains loaded in memory.
- On CPU, responses will be slower — consider using a GPU or quantized model for production.

Security & Limits:
- This simple server does not perform authentication or rate-limiting; avoid exposing it publicly without adding protections.
- No strong validation is performed on uploaded files beyond attempting to open with PIL. You can add file-type checks and size limits.

Deployment suggestions:
- Use a WSGI server (Gunicorn/Waitress) and put the app behind a reverse proxy (Nginx) for production.
- Serve static files via a CDN or from Nginx to offload Flask.

Extending the UI:
- You can add class images to `static/images/` and reference them in `templates/index.html` for a local sample gallery.
- To enable live training or model updates, add an admin-only endpoint that accepts new weight uploads and reloads the model safely.

Troubleshooting
- Model not found: ensure `vegetable_classifier.pth` is in the project root or update `web_app.py` model_path.
- Port conflicts: Flask runs on port 5000 (configurable). If you keep `app.py` Streamlit, Streamlit uses 8501 by default — both can run concurrently on different ports.

Credits
- Model architecture: MobileNetV2 (via `torchvision`), training scripts adapted for this dataset.
- Sample images on `Home` are sourced via Unsplash image URLs for demo purposes.

License
- This repo does not include an explicit license file. Add one (e.g., MIT) if you intend to open-source the project.

---

If you'd like, I can also:
- add a `README` section with example cURL commands for `/predict`;
- remove `app.py` and the `streamlit` dependency if you want to fully switch to the Flask UI;
- add a small `static/images/` folder with 6 local sample photos and reference them in `index.html`.

File: [README.md](README.md)

---

## Code folder structure (detailed)

This section expands the short project structure above and explains the purpose of key files and folders.

- `web_app.py` — Flask server and inference API. Entrypoint for the web UI when using the HTML frontend.
  - `load_model()` builds the architecture via `get_model()` (in `model.py`) and loads weights from `vegetable_classifier.pth`.
  - Change `model_path` here if you store weights elsewhere.

- `app.py` — Streamlit alternative UI. If you prefer the HTML dashboard, you can remove this file and the `streamlit` dependency.

- `model.py` — Defines model architecture (function `get_model(num_classes)`). Keep this synchronized with the architecture used for training.

- `train.py` — Training script. Typical responsibilities:
  - Build dataloaders (using utilities in `data_loader.py`).
  - Construct model via `get_model()` and move to `cuda` if available.
  - Define optimizer, loss, scheduler, and checkpoint saving logic.
  - Save best model to `vegetable_classifier.pth` or other path.

- `data_loader.py` — Dataset utilities. Look here to adjust image transforms, augmentations, or dataset paths.

- `confusion_matrix.py` — Evaluation/visualization helpers.

- `Vegetable Images/` — Expected dataset root when training locally. Under this folder you should have one folder per class (e.g., `Tomato`, `Carrot`, ...). Each class folder contains image files.

- `templates/` and `static/` — Frontend resources served by Flask. `templates/index.html` is the main page; `static/` contains CSS/JS and optional images.

If you add new modules (for example `utils/`, `scripts/`, or experiments), place them under descriptive folders and update `README.md` accordingly.

## Adding or updating the dataset — workflow

Follow these steps when you want to add a new dataset or swap in different images. The examples assume a local dataset and training from scratch or fine-tuning.

1. Download the dataset
  - You already have a dataset link recorded above. Download and extract it locally.

2. Prepare the folder layout (one class per folder)
  - Create a root folder (recommended name: `Vegetable Images/`), then inside create one directory per class label. Example:

```
Vegetable Images/
├─ Tomato/
│  ├─ img001.jpg
│  ├─ img002.jpg
│  └─ ...
├─ Carrot/
├─ Broccoli/
└─ ...
```

Notes:
- Keep class folder names identical to the names expected by the code (the `class_names` list in `web_app.py` uses underscores for multi-word names like `Bitter_Gourd`). Either update `web_app.py` or rename folders to match.

3. Create train/validation/test splits
  - You can organize three subfolders (`train/`, `validation/`, `test/`) with the same class subfolders, or keep a single class folder and use a DataLoader that performs splitting.
  - Example Python script to split images into `train/`, `val/`, and `test/` folders (place in `scripts/split_dataset.py`):

```python
import os, shutil, random
from sklearn.model_selection import train_test_split

src_root = 'Vegetable Images'  # source with one folder per class
dst_root = 'dataset_split'    # output folder
os.makedirs(dst_root, exist_ok=True)

for cls in os.listdir(src_root):
   src_cls = os.path.join(src_root, cls)
   imgs = [os.path.join(src_cls, f) for f in os.listdir(src_cls) if f.lower().endswith(('.jpg','.png'))]
   train, temp = train_test_split(imgs, test_size=0.3, random_state=42)
   val, test = train_test_split(temp, test_size=0.5, random_state=42)

   for split_name, split_list in [('train',train), ('validation',val), ('test',test)]:
      out_dir = os.path.join(dst_root, split_name, cls)
      os.makedirs(out_dir, exist_ok=True)
      for src in split_list:
        shutil.copy(src, out_dir)

print('Done')
```

4. Update `data_loader.py` paths and transforms
  - Open `data_loader.py` and set the dataset path variable to point to `dataset_split/train` (and `validation`/`test` where appropriate).
  - Adjust image transforms if you want different sizes or augmentations. The current inference pipeline expects 224x224 inputs.

5. Update class names in code (if adding a new class)
  - If you add or remove classes, update the `class_names` list in `web_app.py` (and in any training configs) to match the folder names or to map folder names to display names.

6. Train or fine-tune the model
  - Example training command (adapt to `train.py` arguments):

```powershell
python train.py --data dataset_split --epochs 20 --batch-size 32 --save-path vegetable_classifier.pth
```

7. Validate and save weights
  - After training, ensure `vegetable_classifier.pth` is saved in project root (or update `web_app.py` model path accordingly).

8. Test inference locally
  - Start the Flask app:

```powershell
python web_app.py
```

  - Open `http://localhost:5000`, go to `Predict`, upload an image and confirm predictions look correct.

9. Adding a new class later
  - Add a new folder with images under `Vegetable Images/`.
  - Re-run the split script or add images to appropriate split folder.
  - Update `class_names` and retrain or fine-tune the model (recommended: fine-tune from existing weights to save time).

### Tips for good dataset hygiene
- Keep filenames unique and avoid corrupt images. Use a small script to validate image files (open with PIL).
- Balance classes when possible, or use augmentation for under-represented classes.
- Keep a `README` inside the dataset folder describing source and license of images.

---

If you want, I can also:
- add the split script as `scripts/split_dataset.py` in the repo,
- add a small sample `static/images/` folder with example photos, or
- update `data_loader.py` to include a configurable `DATA_ROOT` variable and command-line args.

