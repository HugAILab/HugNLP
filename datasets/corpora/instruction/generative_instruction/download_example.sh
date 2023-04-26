#!/bin/bash

function main() {

    echo "downloading instruction corpora."
    wget https://huggingface.co/datasets/wjn1996/hugnlp-instruction-corpora/resolve/main/instruction_corpora.json.zip
    unzip instruction_corpora.json.zip

}

main "$@"
