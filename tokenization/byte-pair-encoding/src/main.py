# https://huggingface.co/learn/llm-course/chapter6/5

import logging
from collections import defaultdict

from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def create_corpus() -> list[str]:
    """Create sample corpus for training BPE tokenizer."""
    return [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]


def build_word_frequencies(
    corpus: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> defaultdict[str, int]:
    """Build word frequency dictionary from corpus."""
    word_freqs: defaultdict[str, int] = defaultdict(int)
    for text in corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
            text,
        )
        new_words: list[str] = [word for word, _ in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1
    return word_freqs


def build_alphabet(word_freqs: defaultdict[str, int]) -> list[str]:
    """Build alphabet from word frequencies."""
    alphabet: list[str] = []
    for word in word_freqs:
        for letter in word:
            if letter not in alphabet:
                alphabet.append(letter)
    alphabet.sort()
    return alphabet


def compute_pair_freqs(
    splits: dict[str, list[str]],
    word_freqs: defaultdict[str, int],
) -> defaultdict[tuple[str, str], int]:
    """Compute frequency of character pairs."""
    pair_freqs: defaultdict[tuple[str, str], int] = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair: tuple[str, str] = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


def find_best_pair(
    pair_freqs: defaultdict[tuple[str, str], int],
) -> tuple[tuple[str, str], int | None]:
    """Find the most frequent pair."""
    best_pair: tuple[str, str] = ("", "")
    max_freq: int | None = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    return best_pair, max_freq


def merge_pair(
    a: str,
    b: str,
    splits: dict[str, list[str]],
    word_freqs: defaultdict[str, int],
) -> dict[str, list[str]]:
    """Merge the best pair in all splits."""
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = [*split[:i], a + b, *split[i + 2 :]]
            else:
                i += 1
        splits[word] = split
    return splits


def train_bpe_tokenizer(
    corpus: list[str],
    vocab_size: int = 50,
) -> tuple[PreTrainedTokenizerBase, dict[tuple[str, str], str], list[str]]:
    """Train a BPE tokenizer on the given corpus."""
    logger.info("Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    logger.info("Building word frequencies...")
    word_freqs = build_word_frequencies(corpus, tokenizer)
    logger.info(f"Word frequencies: {dict(word_freqs)}")

    logger.info("Building alphabet...")
    alphabet = build_alphabet(word_freqs)
    logger.info(f"Alphabet: {alphabet}")

    # Initialize vocabulary and splits
    vocab: list[str] = ["<|endoftext|>", *alphabet.copy()]
    splits: dict[str, list[str]] = {word: list(word) for word in word_freqs}

    logger.info("Computing initial pair frequencies...")
    pair_freqs = compute_pair_freqs(splits, word_freqs)

    # Show first few pairs
    logger.info("Top pairs:")
    for i, (key, freq) in enumerate(pair_freqs.items()):
        if i >= 5:
            break
        logger.info(f"{key}: {freq}")

    # Find best pair
    best_pair, max_freq = find_best_pair(pair_freqs)
    logger.info(f"Best initial pair: {best_pair} with frequency {max_freq}")

    # Initialize merges dictionary
    merges: dict[tuple[str, str], str] = {("Ġ", "t"): "Ġt"}
    vocab.append("Ġt")

    # Perform initial merge
    splits = merge_pair("Ġ", "t", splits, word_freqs)
    logger.info(
        f"After initial merge - 'Ġtrained': {splits.get('Ġtrained', 'Not found')}",
    )

    logger.info(f"Training BPE to vocabulary size {vocab_size}...")
    # Main BPE training loop
    while len(vocab) < vocab_size:
        pair_freqs = compute_pair_freqs(splits, word_freqs)
        best_pair, max_freq = find_best_pair(pair_freqs)

        if not best_pair[0]:  # Check if best_pair is empty
            logger.warning("No more pairs to merge")
            break

        splits = merge_pair(*best_pair, splits, word_freqs)
        merged_token: str = best_pair[0] + best_pair[1]
        merges[best_pair] = merged_token
        vocab.append(merged_token)

        logger.info(f"Merged {best_pair} -> {merged_token} (freq: {max_freq})")

    logger.info(f"Training complete! Final vocabulary size: {len(vocab)}")
    return tokenizer, merges, vocab


def tokenize_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    merges: dict[tuple[str, str], str],
) -> list[str]:
    """Tokenize text using trained BPE merges."""
    pre_tokenize_result = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
        text,
    )
    pre_tokenized_text: list[str] = [word for word, offset in pre_tokenize_result]
    splits: list[list[str]] = [list(word) for word in pre_tokenized_text]

    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            new_split = []
            i = 0
            while i < len(split):
                if (
                    i < len(split) - 1
                    and split[i] == pair[0]
                    and split[i + 1] == pair[1]
                ):
                    new_split.append(merge)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            splits[idx] = new_split

    return [token for split in splits for token in split]


def main() -> None:
    logger.info("Starting BPE tokenization demonstration")

    # Create corpus
    corpus: list[str] = create_corpus()
    logger.info(f"Created corpus with {len(corpus)} sentences")

    # Train BPE tokenizer
    tokenizer, merges, vocab = train_bpe_tokenizer(corpus, vocab_size=50)

    # Convert merges to regular dict for logging (first 10 items)
    merges_dict: dict[str, str] = {str(k): v for k, v in list(merges.items())[:10]}
    logger.info(f"Final merges (first 10): {merges_dict}")
    logger.info(f"Final vocabulary size: {len(vocab)}")
    logger.info(f"Sample vocab tokens: {vocab[:20]}")  # Show first 20

    # Test tokenization
    test_text: str = "This is not a token."
    tokens: list[str] = tokenize_text(test_text, tokenizer, merges)
    logger.info(f"Tokenized '{test_text}': {tokens}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
