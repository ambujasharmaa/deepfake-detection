# Deepfake Detection

Video deepfake detection on FaceForensics++ (C23) using a fine-tuned EfficientNetB0.

## Pipeline
1. Download FF++ C23 from Kaggle (credentials via Colab Secrets: `KAGGLE_USERNAME`, `KAGGLE_KEY`).
2. Identity-aware train/val/test split (no source/target leakage between splits).
3. Extract 10 face-cropped frames per video (OpenCV Haar cascade, 30% margin).
4. Two-phase transfer learning: frozen backbone, then fine-tune the top layers at a low learning rate.
5. Threshold selected on validation; evaluation at frame level and video level (mean frame probability), with per-manipulation-method recall.

## Run
Open `deepfake_detection.ipynb` in Google Colab with a GPU runtime and run all cells.
