import random


def sample_first(records: list[dict], n: int = 5) -> list[dict]:
    return records[:n]


def sample_last(records: list[dict], n: int = 5) -> list[dict]:
    return records[-n:]


def sample_random(records: list[dict], n: int = 5, seed: int = 42) -> list[dict]:
    if len(records) <= n:
        return records[:]
    return random.Random(seed).sample(records, n)
