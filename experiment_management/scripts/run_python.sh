#!/bin/bash
CONFIG_PATH=config/model/model01/config01.yml
PYTHON_SCRIPT=src/model/model01/image_classifier_models.py
env PYTHONPATH=$(pwd) poetry run python $PYTHON_SCRIPT --config_path $CONFIG_PATH