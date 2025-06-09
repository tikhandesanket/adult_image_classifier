---
tags:
- image-classification
- keras
- adult-content-detection
- computer-vision
- nsfw
license: mit
language: en
datasets: custom
metrics:
- accuracy
---

# Adult Image Classifier

This project contains a deep learning model to classify images as **Adult** or **Non-Adult**.
...

# Adult Image Classifier

This project contains a deep learning model to classify images as **Adult** or **Non-Adult**.

## Features

* Real-time image classification using a CNN model.
* Simple Python script for predicting images one by one or in batch from a folder.
* Uses TensorFlow and Keras.
* Input image size: 128x128 pixels.

## Requirements

* Python 3.x
* TensorFlow
* NumPy
* Pillow (for image processing)

Install dependencies with:

```bash
pip install tensorflow numpy pillow
```

## Usage

1. Download or clone this repository.
2. Download the pre-trained model file (`adult_image_classifier.h5`) from the Hugging Face link below.
3. Place your test images in a folder (e.g., `test_images`).
4. Run the prediction script:

```bash
python predict_images.py
```

## Download Pre-trained Model

The model is hosted on Hugging Face Hub and can be downloaded or loaded directly from:

[Adult Image Classifier on Hugging Face](https://huggingface.co/SanketSanky/adult_image_classifier)

### Download using `wget`:

```bash
wget https://huggingface.co/SanketSanky/adult_image_classifier/resolve/main/adult_image_classifier.h5
```

### Load model directly in Python:

```python
import os
from tensorflow.keras.models import load_model
import requests

model_url = "https://huggingface.co/SanketSanky/adult_image_classifier/resolve/main/adult_image_classifier.h5"
model_path = "adult_image_classifier.h5"

if not os.path.exists(model_path):
    r = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(r.content)

model = load_model(model_path)
```

## Script Overview

* `predict_images.py`: Loads the model and predicts images in a folder.
* `adult_image_classifier.h5`: Pre-trained model file.

## Example Output

```
A.jpg: Non-Adult
nude_man2.jpg: Adult
test2.jpeg: Adult
...
```

## License

MIT License

Sanket tikahnde AI-ML Engineer
