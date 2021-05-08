python3 -u bert_model-BiGRU.py --cuda  --test-eval \
--test-batch-size 1024 \
--test-eval-filename  Test20.csv \
--bert-model data/bert-base-uncased.tar.gz \
--bert-vocab  data/bert-base-uncased-vocab.txt \
--fine-tuned-bert-model output/Causaly_bert-base+BiGRU20_seed_84977986_last_epoch \
--threshold 0.51
