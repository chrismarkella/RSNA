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
python preprocess.py
python model.py
```

## Formating

```bash
pip install yapf
yapf --in-place **/*.py
```