.PHONY: install mlflow train smoke

install:
	python3 -m pip install -U pip
	python3 -m pip install -r requirements.txt

mlflow:
	mlflow ui --backend-store-uri "file:./mlruns" --host 127.0.0.1 --port 5000

smoke:
	python3 -m src.smoke_test

train:
	python3 -m src.train
