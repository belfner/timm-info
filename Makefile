.PHONY: build test publish publish-test clean

build:
	uv build

test:
	pytest -n auto

publish: clean build
	@set -a && . ./.env && set +a && uv publish

publish-test: clean build
	@set -a && . ./.env && set +a && uv publish \
		--publish-url https://test.pypi.org/legacy/ \
		--token "$$UV_PUBLISH_TOKEN_TESTPYPI"

clean:
	rm -rf dist/
