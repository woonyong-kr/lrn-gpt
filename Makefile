PYTHON := .venv/bin/python
export OMP_NUM_THREADS := 1
.PHONY: setup demo test train data train-nsmc
setup:
	uv venv --python 3.12 .venv --allow-existing
	uv pip sync --python $(PYTHON) requirements.lock
test:
	$(PYTHON) -m pytest -q
demo:
	$(PYTHON) -m src.runnable demo
train:
	$(PYTHON) -m src.runnable train

data:
	$(PYTHON) download_data.py
train-nsmc:
	$(PYTHON) -m src.runnable train --config configs/nsmc.json --train data/nsmc_lm_train.txt --validation data/nsmc_lm_val.txt --steps 1000 --checkpoint .artifacts/nsmc.pt
