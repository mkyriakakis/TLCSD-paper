from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import  BertPreTrainedModel, BertModel
import torch
import pandas as pd
import os
from tqdm import tqdm, trange
import random
import time
import numpy as np
from sklearn.metrics import auc,precision_recall_curve

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.optimization import BertAdam




class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            if args["cuda"]:
                loss_fct = torch.nn.BCEWithLogitsLoss()
            else:
                loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
            return logits, loss
        else:
            return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


args = {

    "data_dir": "data",
    "output_dir": "output",
    "bert_model": "data/pytorch_biobert.tar.gz",
    "bert_vocab": "data/biobert_vocab.txt",
    "max_seq_length": 128,
    "train": False,
    "threshold":0.51,
    "predict":False,
    "test_eval":True,
    "test_eval_filename": "Test_v3.csv",
    "eval_filename": "Dev_v3.csv",
    "lower_case": True,
    "train_batch_size": 64,
    "dev_batch_size": 64,
    "test_batch_size": 64,
    "fine_tuned_bert_model": "output/finetuned_pytorch_model_epoch_9.bin",
    "learning_rate": 3e-5,
    "num_train_epochs": 20.0,
    "warmup_proportion": 0.1,
    "cuda": True,
    "seed": 1990,
    "gradient_accumulation_steps": 4,

}



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir, data_file_name, size=-1):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()



class MultiLabelTextProcessor(DataProcessor):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels = None


    def get_train_examples(self, data_dir):
        filename = 'Train_v3.csv'
        data_df = pd.read_csv(os.path.join(data_dir, filename))
        data = data_df[['Sentence','Annotated_Causal']]
        return self._create_examples(data, "train")


    def get_dev_examples(self, data_dir, filename):
        data_df = pd.read_csv(os.path.join(data_dir, filename))
        data = data_df[['Sentence','Annotated_Causal']]
        return self._create_examples(data, "dev")


    def get_test_examples(self, data_dir, data_file_name):
        data_df = pd.read_csv(os.path.join(data_dir, data_file_name))
        data = data_df[['Sentence']]
        return self._create_examples(data_df, "test")


    def _create_examples(self, df, set_type, labels_available=True):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, row) in enumerate(df.values):
            text_a = row[0]
            if labels_available:
                label = row[1]
            else:
                label = None
            examples.append(
                InputExample(text_a=text_a, label=label))
        return examples

def convert_examples_to_features(examples, max_seq_length, tokenizer):

    """Loads a data file into a list of `InputBatch`s."""

    count = 0
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
            count +=1

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=float(example.label)))
    #print(count,len(examples))
    return features



def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

# Prepare model
def get_model(model_state_dict=None):
    if model_state_dict:
        model = BertForMultiLabelSequenceClassification.from_pretrained(args['bert_model'], num_labels = num_labels, state_dict=model_state_dict)
    else:
        model = BertForMultiLabelSequenceClassification.from_pretrained(args['bert_model'], num_labels = num_labels)
    return model



def fit(num_epocs=args['num_train_epochs']):

    global_step = 0
    model.train()
    for i_ in tqdm(range(int(num_epocs)), desc="Epoch"):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            logits, loss = model(input_ids, segment_ids, input_mask, label_ids)
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']


            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                lr_this_step = args['learning_rate'] * warmup_linear(global_step/t_total, args['warmup_proportion'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        #Do evaluation at the end of eaach epoch
        eval()
        #Save model
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args['output_dir'], "finetuned_pytorch_model_epoch_"+str(i_)+".bin")
        torch.save(model_to_save.state_dict(), output_model_file)

def accuracy_thresh(y_pred:torch.Tensor, y_true:torch.Tensor, thresh:float=0.51, sigmoid:bool=True):

    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid: y_pred = y_pred.sigmoid().squeeze(1)

    return ((y_pred>thresh)==y_true.byte()).float().cpu().numpy().sum()


def metrics(y_pred:torch.Tensor, y_true:torch.Tensor, thresh:float=0.51, eps:float=1e-9, sigmoid:bool=True):

    "Computes macro precision, Recall, F1  between `preds` and `targets`"
    if sigmoid: y_pred = y_pred.sigmoid().squeeze(1)
    y_pred = (y_pred>thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum()
    prec = TP/(y_pred.sum()+eps)
    rec = TP/(y_true.sum()+eps)
    f1 = (2*(prec*rec))/(prec+rec+eps)
    return f1, prec, rec


def eval():

    eval_features = convert_examples_to_features(
        eval_examples, args['max_seq_length'], tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args['dev_batch_size'])

    all_logits = None
    all_labels = None

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for idx, batch in enumerate(tqdm(eval_dataloader,desc=" Eval Iteration")):

        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            logits, tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)

        tmp_eval_accuracy = accuracy_thresh(logits, label_ids)
        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

        if all_labels is None:
            all_labels = label_ids.detach().cpu().numpy()
        else:
            all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)


        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    f1 ,pr, rc = metrics(torch.tensor(all_logits),torch.tensor(all_labels))

    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              'eval_f1': f1.numpy().item(),
              'eval_precision': pr.numpy().item(),
              'eval_recall': rc.numpy().item() }

    output_eval_file = os.path.join(args['output_dir'], "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        for key in sorted(result.keys()):
            writer.write("%s = %s\n" % (key, str(result[key])))
        writer.write("--------------------------------------------\n")
    return result

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() and args["cuda"]  else "cpu")
    n_gpu = torch.cuda.device_count()
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args['seed'])
    processor = MultiLabelTextProcessor(args['data_dir'])

    num_labels = 1
    tokenizer = BertTokenizer.from_pretrained(args['bert_vocab'], do_lower_case=args['lower_case'])

    if args['train']:
        print("Pre-process train data ....")
        start = time.time()
        train_examples = processor.get_train_examples(args['data_dir'])
        print('===> Pre-processed train data in: {t:.2f}s'.format(t=time.time()-start))
        num_train_steps = int(len(train_examples) / args['train_batch_size'] / args['gradient_accumulation_steps'] * args['num_train_epochs'])



        print("Load model ....")
        start = time.time()
        model = get_model()
        model.to(device)
        print('===> Finished loading pretrained bert model in: {t:.2f}s'.format(t=time.time()-start))

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        t_total = num_train_steps
        optimizer = BertAdam(optimizer_grouped_parameters,lr=args['learning_rate'],warmup=args['warmup_proportion'],t_total=t_total)

        #Load training data
        print("Creating train data loader ....")
        start = time.time()
        train_features = convert_examples_to_features(train_examples, args['max_seq_length'], tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args['train_batch_size'])
        print('===> Created train dataloader: {t:.2f}s'.format(t=time.time()-start))


        model.unfreeze_bert_encoder()
        # Load eval data
        eval_examples = processor.get_dev_examples(args['data_dir'],args["eval_filename"])
        fit()

    if args["test_eval"]:

        print("Pre-process test data ....")
        start = time.time()
        test_examples = processor.get_dev_examples(args['data_dir'],args["test_eval_filename"])
        print('===> Pre-processed test data in: {t:.2f}s'.format(t=time.time()-start))

        print("Load model ....")
        start = time.time()
        model_state_dict = torch.load(args["fine_tuned_bert_model"])
        model = get_model(model_state_dict)
        model.to(device)
        print('===> Finished loading finetuned bert model in: {t:.2f}s'.format(t=time.time()-start))

        print("Creating test data loader ....")
        start = time.time()
        test_features = convert_examples_to_features(test_examples, args['max_seq_length'], tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.float)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args['test_batch_size'])
        print('===> Created test dataloader: {t:.2f}s'.format(t=time.time()-start))

        all_logits = None
        all_labels = None
        model.eval()

        nb_eval_examples, eval_accuracy = 0, 0
        for step, batch in enumerate(tqdm(test_dataloader, desc="Prediction Iteration")):
            input_ids, input_mask, segment_ids, label_ids = batch
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)
                logits = logits.sigmoid()

            if all_logits is None:
                all_logits = logits.detach().cpu().numpy()
            else:
                all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

            if all_labels is None:
                all_labels = label_ids.detach().cpu().numpy()
            else:
                all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)


            tmp_eval_accuracy = accuracy_thresh(logits, label_ids)
            if all_logits is None:
                all_logits = logits.detach().cpu().numpy()
            else:
                all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

            if all_labels is None:
                all_labels = label_ids.detach().cpu().numpy()
            else:
                all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)


            eval_accuracy += tmp_eval_accuracy
            nb_eval_examples += input_ids.size(0)

        eval_accuracy = eval_accuracy / nb_eval_examples
        f1, pr, rc = metrics(torch.tensor(all_logits),torch.tensor(all_labels),args["threshold"])
        print("---\nF1-score:\t%.4f\tPrecision:\t%.4f\tRecall:\t%.4f\tAccuracy:\t%.4f" % (f1,pr,rc,eval_accuracy))
