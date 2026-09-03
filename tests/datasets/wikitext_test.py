import pytest
import torch

from trainite.datasets.wikitext import WikiTextTransform


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __call__(
        self,
        text,
        add_special_tokens=False,
        truncation=False,
        max_length=None,
    ):
        ids = [10, 11, 12]

        if max_length is not None:
            ids = ids[:max_length]

        return {"input_ids": ids}


def test_wikitext_transform():
    tokenizer = FakeTokenizer()
    transform = WikiTextTransform(tokenizer=tokenizer, max_length=8)

    datapoint = transform({"text": "hello world"})

    assert datapoint.source == "hello world"
    assert datapoint.target == "hello world"

    assert torch.equal(
        datapoint.train_input_ids,
        torch.tensor([1, 10, 11, 12]),
    )

    assert torch.equal(
        datapoint.train_label_ids,
        torch.tensor([10, 11, 12, 2]),
    )

    assert torch.equal(
        datapoint.attention_mask,
        torch.ones(4, dtype=torch.long),
    )

    assert torch.equal(
        datapoint.eval_input_ids,
        torch.tensor([1, 10, 11, 12]),
    )


def test_wikitext_transform_truncates_to_max_length():
    tokenizer = FakeTokenizer()
    transform = WikiTextTransform(tokenizer=tokenizer, max_length=3)

    datapoint = transform({"text": "hello world"})

    assert torch.equal(
        datapoint.train_input_ids,
        torch.tensor([1, 10, 11]),
    )

    assert torch.equal(
        datapoint.train_label_ids,
        torch.tensor([10, 11, 2]),
    )

    assert torch.equal(
        datapoint.attention_mask,
        torch.ones(3, dtype=torch.long),
    )


def test_wikitext_transform_rejects_small_max_length():
    tokenizer = FakeTokenizer()

    with pytest.raises(ValueError, match="at least 2"):
        WikiTextTransform(tokenizer=tokenizer, max_length=1)


def test_wikitext_transform_requires_bos_token():
    tokenizer = FakeTokenizer()
    tokenizer.bos_token_id = None

    transform = WikiTextTransform(tokenizer=tokenizer)

    with pytest.raises(ValueError, match="bos_token_id"):
        transform({"text": "hello world"})


def test_wikitext_transform_requires_eos_token():
    tokenizer = FakeTokenizer()
    tokenizer.eos_token_id = None

    transform = WikiTextTransform(tokenizer=tokenizer)

    with pytest.raises(ValueError, match="eos_token_id"):
        transform({"text": "hello world"})
