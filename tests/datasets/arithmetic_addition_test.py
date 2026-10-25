import pytest
from trainite.datasets.arithmetic_addition import ArithmeticAdditionDataset


def test_train_val_disjoint():
    train = ArithmeticAdditionDataset(split="train")
    val = ArithmeticAdditionDataset(split="val")

    train_sources = {item["source"] for item in train}
    val_sources = {item["source"] for item in val}
    assert train_sources.isdisjoint(val_sources)


def test_split_sizes():
    train = ArithmeticAdditionDataset(split="train")
    val = ArithmeticAdditionDataset(split="val")

    assert len(train) == 10000
    assert len(val) == 1000


def test_sample_format():
    dataset = ArithmeticAdditionDataset(size=100, split="test")
    item = dataset[0]

    assert set(item) == {"source", "target"}
    assert "+" in item["source"]
    a, b = item["source"].split("+")
    assert item["target"] == str(int(a) + int(b))


def test_size_mismatch_raises():
    with pytest.raises(ValueError, match="size must be 10000"):
        ArithmeticAdditionDataset(size=1000, split="train")

    with pytest.raises(ValueError, match="size must be 1000"):
        ArithmeticAdditionDataset(size=100, split="val")


def test_test_split_independent_generation():
    test = ArithmeticAdditionDataset(size=1000, min_digits=2, max_digits=4, seed=200, split="test")
    train = ArithmeticAdditionDataset(split="train")

    test_sources = [item["source"] for item in test]
    assert len(test) == 1000
    assert len(set(test_sources)) == 1000
    assert test_sources != [item["source"] for item in train][:1000]


def test_test_split_disjoint_by_digit_range():
    test = ArithmeticAdditionDataset(size=1000, min_digits=5, max_digits=5, seed=200, split="test")
    train = ArithmeticAdditionDataset(split="train")

    assert len(test) == 1000
    test_sources = {item["source"] for item in test}
    train_sources = {item["source"] for item in train}
    assert test_sources.isdisjoint(train_sources)
    assert all(len(s.split("+")[0]) == 5 and len(s.split("+")[1]) == 5 for s in test_sources)


def test_same_seed_reproducible():
    a = ArithmeticAdditionDataset(split="train")
    b = ArithmeticAdditionDataset(split="train")

    assert [item["source"] for item in a] == [item["source"] for item in b]


def test_invalid_split_raises():
    with pytest.raises(ValueError, match="split must be"):
        ArithmeticAdditionDataset(split="dev")
