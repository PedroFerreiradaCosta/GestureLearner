#!/bin/bash
printf '%s\n' --------------------
echo ARGS
printf '%s\n' --------------------

experiment=${1}


printf '%s\n' --------------------
echo ENV
printf '%s\n' --------------------
export MLFLOW_TRACKING_URI=file:/project/mlruns
export MLFLOW_EXPERIMENT_NAME=${experiment}

echo $MLFLOW_EXPERIMENT_NAME
echo $MLFLOW_TRACKING_URI

printf '%s\n' --------------------
echo PIP
printf '%s\n' --------------------
pip install -r /project/requirements.txt
echo SUCCESS

printf '%s\n' --------------------
echo PYTHON
printf '%s\n' --------------------
python3 /convnet/code/train.py
python3 /convnet/code/test.py