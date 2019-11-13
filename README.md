RSNA
=====

RSNA Intracranial Hemorrhage Detection

## Setup

```bash
git clone https://github.com/chrismarkella/RSNA
cd RSNA
# If you don't have poetry, please install it here:
# https://github.com/sdispater/poetry
poetry install
poetry shell
python preprocess.py
python model.py
```