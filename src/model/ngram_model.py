import json
import os
from collections import defaultdict, Counter


class NGramModel:
    """
    Responsible for building, storing, saving, loading,
    and performing backoff lookup on n-gram probability tables.
    """

    def __init__(self, n: int):
        self.n = n
        self.vocab = set()
        self.counts = {i: defaultdict(Counter) for i in range(1, n + 1)}
        self.probs = {}

    def build_vocab(self, token_file: str, unk_threshold: int) -> None:
        word_counts = Counter()

        with open(token_file, encoding="utf-8") as f:
            for line in f:
                word_counts.update(line.split())

        self.vocab = {w for w, c in word_counts.items() if c >= unk_threshold}
        self.vocab.add("<UNK>")

    def build_counts_and_probabilities(self, token_file: str) -> None:
        with open(token_file, encoding="utf-8") as f:
            for line in f:
                words = [
                    w if w in self.vocab else "<UNK>"
                    for w in line.split()
                ]

                for i in range(len(words)):
                    for order in range(1, self.n + 1):
                        if i + order <= len(words):
                            ngram = words[i:i + order]
                            context = tuple(ngram[:-1])
                            word = ngram[-1]
                            self.counts[order][context][word] += 1

        # Convert counts → probabilities
        self.probs = {}

        for order, table in self.counts.items():
            self.probs[order] = {}
            for context, counter in table.items():
                total = sum(counter.values())
                self.probs[order][context] = {
                    w: c / total for w, c in counter.items()
                }

    def lookup(self, context: list[str]) -> dict:
        """
        Backoff lookup from n-gram down to unigram.
        """
        for order in range(self.n, 0, -1):
            if order > 1:
                ctx = context[-(order - 1):]
                ctx = tuple(ctx)
            else:
                ctx = ()

            if ctx in self.probs.get(order, {}):
                return self.probs[order][ctx]

        return {}

    def save_model(self, path: str) -> None:
        """
        Save model probabilities to JSON using string contexts.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        serializable = {}

        for order, table in self.probs.items():
            serializable[f"{order}gram"] = {}
            for context, next_words in table.items():
                context_str = " ".join(context)
                serializable[f"{order}gram"][context_str] = next_words

        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    def save_vocab(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(sorted(self.vocab), f, indent=2)

    def load(self, model_path: str, vocab_path: str) -> None:
        with open(model_path, encoding="utf-8") as f:
            raw = json.load(f)

        self.probs = {}

        for key, table in raw.items():
            order = int(key.replace("gram", ""))
            self.probs[order] = {}

            for context_str, next_words in table.items():
                if context_str == "":
                    context = ()
                else:
                    context = tuple(context_str.split())

                self.probs[order][context] = next_words

        with open(vocab_path, encoding="utf-8") as f:
            self.vocab = set(json.load(f))