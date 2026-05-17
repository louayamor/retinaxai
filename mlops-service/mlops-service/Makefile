.PHONY: help imaging clinical clean

help:
	@echo "RetinaXAI MLOps Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  imaging         Run imaging pipeline stages"
	@echo "  clinical        Run clinical pipeline stages"
	@echo "  clean           Clean outputs"

imaging:
	python main.py pipeline --stage ingest --pipeline imaging
	python main.py pipeline --stage clean --pipeline imaging
	python main.py pipeline --stage transform --pipeline imaging
	python main.py pipeline --stage train --pipeline imaging
	python main.py pipeline --stage evaluate --pipeline imaging

clinical:
	python main.py pipeline --stage ingest --pipeline clinical
	python main.py pipeline --stage clean --pipeline clinical
	python main.py pipeline --stage transform --pipeline clinical
	python main.py pipeline --stage train --pipeline clinical
	python main.py pipeline --stage evaluate --pipeline clinical

clean:
	rm -rf artifacts/data/processed/*
	rm -rf artifacts/model/*
	rm -f artifacts/training_jobs.json