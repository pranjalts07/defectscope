# DefectScope

**A practical manufacturing defect detection project for AI/ML internship portfolios.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

DefectScope is a compact end-to-end CV system for surface defect detection on MVTec AD. It combines a supervised DenseNet classifier with an unsupervised convolutional autoencoder, then exposes both through a FastAPI service with Grad-CAM explanations.

## Why this project stands out

- It shows full-stack ML thinking, not just model training.
- It uses two complementary approaches: classification and anomaly detection.
- It includes evaluation, threshold calibration, API serving, Docker support, and tests.
- It is easy to explain in an interview: train on MVTec, calibrate thresholds, deploy an inference endpoint, and return an explanation heatmap.

## What it does

1. **DenseNet-121 classifier**: predicts good vs defective and returns confidence.
2. **Convolutional autoencoder**: learns only normal images and flags unusual reconstruction error.
3. **Grad-CAM overlay**: shows which regions influenced the classifier decision.
4. **Review flag**: if the two signals disagree, the request is marked for human review.

## Example response

```bash
curl -X POST http://localhost:8000/predict \
      -F "file=@test_bottle.png" | python -m json.tool
```

```json
{
      "prediction": "defective",
      "confidence": 0.9712,
      "anomaly_score": 0.024831,
      "anomaly_threshold": 0.012,
      "needs_review": false,
      "heatmap_b64": "iVBORw0KGgo...",
      "latency_ms": 38.4
}
```

## Results on bottle category

| Model | AUC-ROC | F1 | Precision | Recall | Latency |
|---|---|---|---|---|---|
| DenseNet-121 | 0.984 | 0.962 | 0.929 | 1.000 | ~40 ms |
| Autoencoder | 0.649 | 0.863 | 0.759 | 1.000 | ~15 ms |

The important part for interviews is not the raw metric table. It is the story: the classifier is the primary signal, the autoencoder is a safety net for novel defects, and threshold calibration matters because the dataset is imbalanced.

## Architecture

```text
Image
      -> preprocess
      -> DenseNet classifier -> class + confidence
      -> Autoencoder -> reconstruction error
      -> threshold logic -> needs_review
      -> optional Grad-CAM heatmap
```

## Local setup

```bash
python -m venv defectscope-env
source defectscope-env/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Download MVTec AD to `data/raw/`, then verify the structure:

```bash
python scripts/download_mvtec.py --data_dir data/raw
```

Train and evaluate:

```bash
python -m training.train_cnn --category bottle
python -m training.train_autoencoder --category bottle
python -m evaluation.threshold_search --category bottle
python -m evaluation.evaluate --category bottle
```

Run the API:

```bash
uvicorn api.main:app --reload --port 8000
```

Or with Docker:

```bash
docker-compose up --build
```

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict` | POST | Upload an image and get a defect prediction |
| `/health` | GET | Check whether the models loaded successfully |
| `/metrics` | GET | View recent request counts and latency |
| `/threshold/update` | POST | Update the anomaly threshold at runtime |

OpenAPI docs are available at `http://localhost:8000/docs` once the service is running.

## What to say in an internship interview

- Built a dual-model defect detection pipeline for manufacturing QA.
- Handled preprocessing, model loading, calibration, and explainability.
- Shipped the model behind a FastAPI endpoint with Docker support.
- Added tests so the project can be run without the full dataset.

## Tests

```bash
pytest tests/ -v
```

The tests cover model forward passes, mocked API endpoints, and dataset loading logic.

## Honest limitations

- It currently targets one MVTec category at a time.
- The autoencoder is a secondary signal, not a perfect anomaly detector.
- Thresholds need calibration per category.

## Next improvements

- ONNX export for faster CPU inference
- Multi-category training
- Pixel-level anomaly localization from reconstruction error maps
- Small demo UI for non-technical users

## References

- [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [Grad-CAM](https://arxiv.org/abs/1610.02391)
- [DenseNet](https://arxiv.org/abs/1608.06993)

*Built as a portfolio project to show practical ML engineering, not just notebook experimentation.*
