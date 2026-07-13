import random


def make_holdout_split(cameras, holdout_ratio=0.0, seed=2026):
    if holdout_ratio <= 0:
        return list(range(len(cameras))), []
    if holdout_ratio >= 1:
        raise ValueError("holdout_ratio must be < 1.0")

    indices = list(range(len(cameras)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    holdout_count = max(1, int(round(len(indices) * holdout_ratio)))
    if holdout_count >= len(indices):
        holdout_count = len(indices) - 1

    val_indices = sorted(indices[:holdout_count])
    train_indices = sorted(indices[holdout_count:])
    return train_indices, val_indices


def image_names_for_indices(cameras, indices):
    return {cameras[i]["image_name"] for i in indices}
