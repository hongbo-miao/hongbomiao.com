#!/usr/bin/env bash
set -e

python export_onnx.py

# Generate a model plan that will be used to host SageMaker Endpoint
trtexec \
  --onnx=model.onnx \
  --saveEngine=model.plan \
  --minShapes=input:1x3x224x224 \
  --optShapes=input:128x3x224x224 \
  --maxShapes=input:128x3x224x224 \
  --explicitBatch \
  --fp16 \
  --verbose \
    | tee conversion.txt
