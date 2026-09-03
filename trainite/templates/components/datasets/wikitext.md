# WikiText dataset

This built-in dataset loads a WikiText dataset configuration from Hugging Face
Datasets and applies a causal language-modeling transform to each sample.

## Configure the dataset

The default configuration uses the WikiText-2 raw dataset:

```yaml
data:
  dataset:
    path: Salesforce/wikitext
    name: wikitext-2-raw-v1
    split: train
```

The available WikiText configurations are:

* `wikitext-103-raw-v1`
* `wikitext-103-v1`
* `wikitext-2-raw-v1`
* `wikitext-2-v1`

WikiText samples contain a single `text` field. The built-in transform uses this field as the
causal language-modeling sequence.

## Configure the transform

The transform uses the tokenizer configured by Trainite and supports:

```yaml
data:
  transform:
    max_length: 128
```

`max_length` controls the maximum sequence length.

The transform adds the tokenizer's BOS and EOS tokens and creates:

* `train_input_ids`
* `train_label_ids`
* `attention_mask`
* `eval_input_ids`

## Dataset splits

WikiText provides `train`, `validation`, and `test` splits. Trainite's data configuration can
create the training/validation split according to the configured ratios.

The default configuration uses:

```yaml
test_ratio: 0.1
val_ratio: 0.1
```

## Hugging Face authentication

The dataset is downloaded and cached by Hugging Face Datasets. Do not put Hugging Face access
tokens in `config.yaml`. Authenticate through the Hugging Face CLI or the environment when required.
