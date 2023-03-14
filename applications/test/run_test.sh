#!/usr/bin/env bash

set -e
export CUDA_VISIBLE_DEVICES=-1

export MKL_THREADING_LAYER=GNU

# Unit tests
echo "================== Test user defined classification =================="

bash applications/test/test_cls_cpu.sh
bash applications/test/test_cls_code_cpu.sh
rm -rf output/

# pip uninstall easynlp -y
