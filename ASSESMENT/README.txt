DATASET SETUP

Final assessment dataset:
New Plant Diseases Dataset (Kaggle)

Download/extract it into:
  data/new_plant_diseases/

Required structure:
  data/new_plant_diseases/train/<disease-class>/*.jpg
  data/new_plant_diseases/valid/<disease-class>/*.jpg

The training script uses 85% of TRAIN for fitting, 15% for validation, and keeps
VALID completely separate as the held-out TEST set. This lets the report compare
actual labels against CNN predictions without using test images for training.

For development only, synthetic/AI-generated images may be used to test the UI
pipeline. They must never be reported as genuine Kaggle samples or used as the
final assessment metrics.
