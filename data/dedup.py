from datasketch import MinHash, MinHashLSH
from loguru import logger

DEFAULT_NUM_PERM = 128
DEFAULT_SHINGLE_SIZE = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.98


def get_minhash(
    text: str, num_perm: int = DEFAULT_NUM_PERM, shingle_size: int = DEFAULT_SHINGLE_SIZE
) -> MinHash:
    """Compute a MinHash signature for a text using word-level shingles.

    Args:
        text: Input text to hash.
        num_perm: Number of permutation functions (higher = more accurate,
            slower).
        shingle_size: Size of word n-grams (shingles) used to build the set
            representation of the text.

    Returns:
        A MinHash object representing the text.
    """
    tokens = text.lower().split()
    shingles = {
        " ".join(tokens[i : i + shingle_size])
        for i in range(max(len(tokens) - shingle_size + 1, 1))
    }
    minhash = MinHash(num_perm=num_perm)
    for shingle in shingles:
        minhash.update(shingle.encode("utf-8"))
    return minhash


def deduplicate_indices(
    texts: list[str], threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> list[int]:
    """Return indices of texts to keep after removing near-duplicates.

    For each cluster of texts with Jaccard similarity >= threshold, only
    the first-seen index is kept.

    Args:
        texts: List of texts to deduplicate.
        threshold: Similarity threshold above which two texts are
            considered duplicates (0.95 = 95% similar).

    Returns:
        Sorted list of indices to keep, in original order.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=DEFAULT_NUM_PERM)
    keep_indices = []

    for i, text in enumerate(texts):
        minhash = get_minhash(text)
        if not lsh.query(minhash):
            lsh.insert(str(i), minhash)
            keep_indices.append(i)

    removed_count = len(texts) - len(keep_indices)
    logger.info(
        f"Deduplication: kept {len(keep_indices)}/{len(texts)} "
        f"({removed_count} near-duplicates removed at threshold={threshold})"
    )
    return keep_indices
