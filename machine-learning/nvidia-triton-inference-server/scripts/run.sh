#!/usr/bin/env bash
set -e

# https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html

TRITON_SERVER_VERSION=24.04

git clone https://github.com/triton-inference-server/server.git
cd server/docs/examples
./fetch_models.sh

# Start Triton server
docker run \
  --gpus=1 \
  --publish=8000:8000 \
  --publish=8001:8001 \
  --publish=8002:8002 \
  --volume="${PWD}/model_repository:/models" \
  --rm \
  "nvcr.io/nvidia/tritonserver:${TRITON_SERVER_VERSION}-py3" \
    tritonserver \
      --model-repository=/models

# Start Triton client
docker run \
  --interactive \
  --tty \
  --rm \
  --network=host \
  "nvcr.io/nvidia/tritonserver:${TRITON_SERVER_VERSION}-py3-sdk"
# /workspace/install/bin/image_client -m densenet_onnx -c 3 -s INCEPTION /workspace/images/mug.jpg
