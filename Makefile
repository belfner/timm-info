.PHONY: build test upload-test upload

build:
	python setup.py sdist bdist_wheel

test:
	pytest -n auto

upload-test:
	python -m twine upload --repository testpypi dist/* --verbose

upload:
	python -m twine upload dist/*
