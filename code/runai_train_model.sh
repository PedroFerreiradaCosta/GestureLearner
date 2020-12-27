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
echo DOWNLOAD REFERENCE SCRIPT
printf '%s\n' --------------------
git clone https://github.com/pytorch/vision.git
cd /project/code/vision
git checkout v0.3.0
cp references/detection/utils.py ../
cp references/detection/transforms.py ../
cp references/detection/coco_eval.py ../
cp references/detection/engine.py ../
cp references/detection/coco_utils.py ../
cd ..

printf '%s\n' --------------------
echo PYTHON
printf '%s\n' --------------------
echo PREPROCCESSING
python3 /project/code/clean_egohands_dataset.py
printf '%s\n' --------------------
echo TRAINING
python3 /project/code/train.py