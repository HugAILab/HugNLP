#!/usr/bin/env bash

set -e
export CUDA_VISIBLE_DEVICES=-1

export MKL_THREADING_LAYER=GNU

# Unit tests
echo "================== Test user defined classification =================="

bash applications/test/test_cls_cpu.sh # test default cls

echo "================== Test user defined labeling =================="

bash applications/test/test_labeling_cpu.sh # test default cls

echo "================== Test user defined code clone =================="

bash applications/test/test_cls_code_cpu.sh # test code cls

echo "================== Test causal lm incontext learning for classification =================="

bash applications/test/test_causal_incontext_cls_cpu.sh # test gpt2 in-context cls

# rm -rf output/

# pip uninstall easynlp -y
