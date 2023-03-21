#!/usr/bin/env bash

set -e
export CUDA_VISIBLE_DEVICES=-1

export MKL_THREADING_LAYER=GNU

# Unit tests
echo "================== Test user defined classification =================="

bash applications/test/test_cls_cpu.sh # test default cls
bash applications/test/test_cls_code_cpu.sh # test code cls
bash applications/test/test_causal_incontext_cls_cpu.sh # test gpt2 in-context cls
rm -rf output/

# pip uninstall easynlp -y
