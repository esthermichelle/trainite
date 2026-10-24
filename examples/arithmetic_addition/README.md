# Arithmetic Addition Experiment

## Objective

Investigate whether a character-level Transformer can learn
arithmetic addition from examples with 2–4 digit operands and
generalize to operands with 5 digits.

## Dataset

The dataset is generated synthetically by
`ArithmeticAdditionDataset`.

Each sample contains:

- `source`: an addition expression, such as `123+456`
- `target`: the correct sum, such as `579`

## Data Splits

| Split | Operand digits | Number of samples |
| --- | --- | ---: |
| Train | 2–4 | 10,000 |
| Validation | 2–4 | 1,000 |
| Test | 5 | 1,000 |
