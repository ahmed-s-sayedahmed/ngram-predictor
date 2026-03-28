import json
from collections import defaultdict, Counter

class NGramModel:
    def __init__(self, n):
        self.n = n
        self.vocab = set()
        self.counts = {i: defaultdict(Counter) for i in range(1, n+1)}
        self.probs = {}

    def build_vocab(self, token_file, unk_threshold):
        word_counts = Counter()

        with open(token_file) as f:
            for line in f:
                word_counts.update(line.split())

        self.vocab = {w for w, c in word_counts.items() if c >= unk_threshold}
        self.vocab.add("<UNK>")

    def build_counts_and_probabilities(self, token_file):
        with open(token_file) as f:
            for line in f:
                words = line.split()

                for i in range(len(words)):
                    for n in range(1, self.n + 1):
                        if i + n <= len(words):
                            ngram = words[i:i+n]
                            context = tuple(ngram[:-1])
                            word = ngram[-1]
                            self.counts[n][context][word] += 1

        # Convert to probabilities
        self.probs = {}
        for n in self.counts:
            self.probs[n] = {}
            for context, counter in self.counts[n].items():
                total = sum(counter.values())
                self.probs[n][context] = {
                    w: c / total for w, c in counter.items()
                }

    def lookup(self, context):
        for order in range(self.n, 0, -1):
            ctx = tuple(context[-(order-1):]) if order > 1 else ()
            if ctx in self.probs.get(order, {}):
                return self.probs[order][ctx]
        return {}

    # def save_model(self, path):
    #     with open(path, "w") as f:
    #         json.dump(self.probs, f)

def load(self, model_path, vocab_path):
    import json

    with open(model_path) as f:
        raw = json.load(f)

    self.probs = {}

    for n in raw:
        order = int(n)
        self.probs[order] = {}

        for context_str, next_words in raw[n].items():
            if context_str == "":
                context = ()
            else:
                context = tuple(context_str.split())

            self.probs[order][context] = next_words

    with open(vocab_path) as f:
        self.vocab = set(json.load(f))


    # def save_vocab(self, path):
    #     with open(path, "w") as f:
    #         json.dump(list(self.vocab), f)