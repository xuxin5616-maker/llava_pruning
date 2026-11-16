"""Pick a reproducible, uniform subset of distinct query images."""

from collections import defaultdict
import random


def sample_images(records, count, seed, image_key):
    by_image = defaultdict(list)
    for record in records:
        by_image[image_key(record)].append(record)
    if not 1 <= count <= len(by_image):
        raise ValueError(f"Requested {count} images, but only {len(by_image)} distinct images are available")
    if not 0 <= seed < 2**32:
        raise ValueError("--sample-seed must be an integer in [0, 2**32)")
    rng = random.Random(seed)
    selected = rng.sample(list(by_image.values()), count)
    return [rng.choice(group) if len(group) > 1 else group[0] for group in selected], len(by_image)
