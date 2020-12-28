experiment="FastRCNN"

runai submit \
  --name pedro-fastrcnn \
  --image 10.202.67.201:32581/wds20:gesturelearning \
  --backoffLimit 0 \
  --gpu 1 \
  --cpu 8 \
  --project wds20 \
  --volume /nfs/home/wds20/pedro/GestureLearner/:/project \
  --command -- bash /project/code/runai_train_model.sh \
  ${experiment}

