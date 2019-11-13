RSNA
====

RSNA Intracranial Hemorrhage Detection

## Setup

```bash
git clone https://github.com/chrismarkella/RSNA
cd RSNA
# If you don't have poetry, please install it here:
# https://github.com/sdispater/poetry
poetry install
# A temporary workaround
pip install tensorflow Pillow
poetry shell

# Extract the images
unzip -Z1 train_images.zip |sort |head -50000|xargs unzip -q train_images.zip

# Preprocess the images and run the model
python preprocess.py
python model.py
```

## Formating

```bash
pip install yapf
yapf --in-place **/*.py
```
