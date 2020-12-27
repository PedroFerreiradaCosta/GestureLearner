
#!/bin/bash
#
# Script to build the Docker image to train the models.
#
# $ create_dockerimage.sh
set -ex
TAG=gesturelearning

docker build --tag "10.202.67.201:32581/${USER}:${TAG}" -f ./Dockerfile . \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  --build-arg USER=${USER}

docker push "10.202.67.201:32581/${USER}:${TAG}"