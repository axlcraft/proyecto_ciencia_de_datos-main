.PHONY: install ingest silver gold holdout train pipeline test lint clean \
	notebooks validar-externa figuras informes

install:
	pip install -e ".[dev]"

ingest:
	python -m flows.job --stage ingestion

silver:
	python -m flows.job --stage silver

gold:
	python -m flows.job --stage gold

holdout:
	python -m flows.08_holdout_evaluation

train:
	python -m flows.job --stage train

pipeline:
	python -m flows.job --stage all

test:
	pytest -q

lint:
	ruff check .

notebooks:
	mkdir -p reports/ejecutados
	python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 \
		--output-dir reports/ejecutados --output 02_exploracion.ejecutado.ipynb \
		notebooks/02_exploracion.ipynb
	python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 \
		--output-dir reports/ejecutados --output 04_modelado.ejecutado.ipynb \
		notebooks/04_modelado.ipynb

validar-externa:
	python -m flows.06_external_validation

figuras:
	python -m flows.07_report_figures

informes: figuras validar-externa

clean:
	rm -rf data models .pytest_cache
