PYTHON ?= python

.PHONY: help install test gate package qualification ci-fast ci-full verify build

help:
	@$(PYTHON) scripts/build_pipeline.py --list

install:
	@$(PYTHON) scripts/build_pipeline.py --stage install

test:
	@$(PYTHON) scripts/build_pipeline.py --stage test

gate:
	@$(PYTHON) scripts/build_pipeline.py --stage gate

package:
	@$(PYTHON) scripts/build_pipeline.py --stage package

qualification:
	@$(PYTHON) scripts/build_pipeline.py --stage qualification

ci-fast:
	@$(PYTHON) scripts/build_pipeline.py --stage ci-fast

ci-full:
	@$(PYTHON) scripts/build_pipeline.py --stage ci-full

verify: test
build: package
