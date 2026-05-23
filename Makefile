serve:
	uvicorn app.main:app --reload --port 8081

test:
	pytest tests/ -v

demo:
	python notebooks/demo.py

seed:
	python scripts/seed_model.py
