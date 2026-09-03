# AgriVision AI — CNN Crop Disease Detection & Validation

Premium black/yellow frontend + TensorFlow CNN + Flask API + real held-out validation.

## 1. Dataset
Use the **New Plant Diseases Dataset (Kaggle)**. Download/extract it into:

```text
data/new_plant_diseases/train/<class>/*.jpg
data/new_plant_diseases/valid/<class>/*.jpg
```

The training script uses 85% of `train` for model fitting and 15% for validation. The separate `valid` directory is kept untouched and evaluated as the held-out **TEST** set.

## 2. Install
From the project root:

```bash
pip install tensorflow flask flask-cors pillow scikit-learn numpy
```

## 3. Train + evaluate

```bash
cd training
python train_cnn.py
```

The script saves:

- `models/crop_disease_cnn.keras` — trained CNN
- `metrics/metrics.json` — accuracy, precision, recall, F1, confusion matrix
- `metrics/training_history.json` — training/validation curves
- `metrics/classes.json` — class order
- `metrics/validation_samples.json` — auditable actual-vs-predicted examples

## 4. Start API
Open another terminal:

```bash
cd backend
python app.py
```

API: `http://127.0.0.1:5000`

## 5. Start website
Open another terminal:

```bash
cd frontend
python -m http.server 5500
```

Open `http://127.0.0.1:5500`.

The website now:

- sends uploaded leaf images to the real CNN through `/predict`;
- displays crop, disease and confidence;
- loads actual evaluation metrics from `/metrics`;
- loads actual-vs-predicted validation evidence from `/validation-samples`;
- displays training/validation accuracy curves.

## 6. Faculty demonstration
Use a labelled image from the held-out `valid` dataset. The folder name is the ground-truth label. Upload that image in the Scanner and compare the website's prediction with its original folder label. The Validation Lab also shows a set of automatically generated comparisons.

**Important:** synthetic/AI-generated images are for UI/pipeline development only. Do not report synthetic metrics as final Kaggle results.
