# This makefile has been created to help developers perform common actions.
# Most actions assume it is operating in a virtual environment where the
# python command links to the appropriate virtual environment Python.

# Do not remove this block. It is used by the 'help' rule when
# constructing the help output.
# help:
# help: masktools Makefile help
# help:

CONDA_BASEDIR := $(shell conda info --base)
# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash

# help: help                           - display this makefile's help information
.PHONY: help
help:
	@grep "^# help\:" Makefile | grep -v grep | sed 's/\# help\: //' | sed 's/\# help\://'

# help: conda                          - create a conda environment for development
.PHONY: conda
conda:
	@conda env create -f conda.dev.yaml --force
	@source ${CONDA_BASEDIR}/bin/activate masktools && python -m pip install -e .

# help: clean                          - clean all files using .gitignore rules
.PHONY: clean
clean:
	@git clean -X -f -d


# help: scrub                          - clean all files, even untracked files
.PHONY: scrub
scrub:
	@git clean -x -f -d


# help: test                           - run tests
.PHONY: test
test:
	@pytest masktools tests


# help: style                          - perform code formatting
.PHONY: style
style:
	@isort -rc -y masktools tests
	@black masktools tests


# help: check                          - perform linting checks
.PHONY: check
check:
	@black --check masktools tests
	@pylint masktools tests
	@mypy -p masktools
	@mypy tests

# help: hooks                          - setup git hooks
.PHONY: hooks
hooks:
	@git config core.hooksPath `pwd`/hooks


# Keep these lines at the end of the file to retain nice help
# output formatting.
# help:
