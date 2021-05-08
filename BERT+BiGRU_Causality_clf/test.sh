#!/bin/bash
for filename in output/CausalTB2/*last_epoch; do
stdbuf -oL python3 -u bert_model-BiGRU.py --cuda --test-eval \
--test-batch-size 128 \
--test-eval-filename  CausalTB_test_original.csv \
--bert-model data/bert-base-uncased.tar.gz \
--bert-vocab  data/bert-base-uncased-vocab.txt \
--fine-tuned-bert-model $filename \
--threshold 0.51 
#| while IFS= read -r line
#do
#tee -a output/EventSL/EventSL_last.txt
#done
done
