# Analyses behind the four poster figures. Run `make all` after `make data`.
PY      ?= .venv/bin/python
export MPLCONFIGDIR ?= /tmp/microstructure-mpl

.PHONY: help setup data derived recall timecourse content viewing-null ceiling figures all

help:
	@echo "make setup        create .venv and install requirements"
	@echo "make data         fetch the Chen et al. release + segment labels (transcripts: see README)"
	@echo "make derived      scene table, movie timeline, invalid-TR scan"
	@echo "make recall       Fig 3 / 4A  recall-BOLD plant identity (source-aligned)"
	@echo "make timecourse   Fig 4B      onset-centred identity time course"
	@echo "make content      Fig 4C      spoken-content associations"
	@echo "make viewing-null Fig 2       matched pseudo-narrative viewing null"
	@echo "make ceiling      Fig 3       order-valid null draws + reliability ceiling"
	@echo "make figures      Fig 1-4     true-size poster PDFs in figures/"
	@echo "make all          everything above except setup/data"

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.fetch_data

derived:
	$(PY) -m src.parse_scenes
	$(PY) -m src.data_quality

recall: derived
	$(PY) -m src.run_chen_timing_corrected

timecourse: recall
	$(PY) -m src.run_recall_timecourse --timing-corrected

content: recall
	$(PY) -m src.run_recall_content

viewing-null: derived
	$(PY) -m src.run_viewing_pseudo_null

ceiling: recall
	$(PY) -m src.export_order_valid_null
	$(PY) -m src.compute_identity_ceiling

figures:
	$(PY) -m src.make_poster_figures

all: recall timecourse content viewing-null ceiling figures
