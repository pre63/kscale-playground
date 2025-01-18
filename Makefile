# Makefile

define HELP_MESSAGE
playground

# Installing

1. Create a new Conda environment: `conda create --name kscale-mujoco-playground python=3.12`
2. Activate the environment: `conda activate kscale-mujoco-playground`
3. Install the package: `make install-dev`

# Running Tests

1. Run autoformatting: `make format`
2. Run static checks: `make static-checks`
3. Run unit tests: `make test`

endef
export HELP_MESSAGE

all:
	@echo "$$HELP_MESSAGE"
.PHONY: all

# ------------------------ #
#          Build           #
# ------------------------ #

install:
	@pip install --verbose -e .
.PHONY: install

install-dev:
	@pip install --verbose -e '.[dev]'
.PHONY: install

build-ext:
	@python setup.py build_ext --inplace
.PHONY: build-ext

clean:
	rm -rf build dist *.so **/*.so **/*.pyi **/*.pyc **/*.pyd **/*.pyo **/__pycache__ *.egg-info .eggs/ .ruff_cache/
.PHONY: clean

# ------------------------ #
#       Static Checks      #
# ------------------------ #


format:
	@isort --profile black playground
	@black playground
	@ruff format playground
	@isort playground
.PHONY: format

static-checks:
	@isort --profile black --check --diff playground
	@black --diff --check playground
	@ruff check playground
	@mypy --install-types --non-interactive playground
.PHONY: lint

mypy-daemon:
	@dmypy run -- $(py-files)
.PHONY: mypy-daemon

# ------------------------ #
#        Unit tests        #
# ------------------------ #

test:
	python -m pytest
.PHONY: test

train:
	python playground/runner.py --env ZbotJoystickFlatTerrain --task flat_terrain --save-model --seed 42 --num-episodes 2 --episode-length 10 --x-vel 2.0 --y-vel 1.0 --yaw-vel 0.5

eval:
	python playground/runner.py --env ZbotJoystickFlatTerrain --task flat_terrain --load-model --seed 42 --num-episodes 1 --episode-length 10 --x-vel 1.5 --y-vel 0.5 --yaw-vel 0.2