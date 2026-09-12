.PHONY: setup prepare train index eval reproduce test demo UI clean

setup:
	pip install -r requirements.txt

prepare:
	python scripts/prepare_data.py --brand AmazonHelp --max-rows 100000

train:
	python scripts/train_models.py

index:
	python scripts/build_index.py

eval:
	python scripts/reproduce_results.py

reproduce:
	python scripts/reproduce_results.py

test:
	pytest tests/

demo:
	python scripts/run_demo.py

ui:
	streamlit run app/streamlit_app.py

clean:
	rm -rf data/processed/* reports/*.csv reports/*.json *.pkl __pycache__
