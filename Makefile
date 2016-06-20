.PHONY: test init

init:
	python -m pip install -r requirements.txt

test:
	cd tests/ && python -m unittest
