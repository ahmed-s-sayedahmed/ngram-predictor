class Predictor:
    def __init__(self, model, normalizer):
        self.model = model
        self.normalizer = normalizer

    def map_oov(self, words):
        return [w if w in self.model.vocab else "<UNK>" for w in words]

    def predict_next(self, text, k):
        text = self.normalizer.normalize(text)
        words = text.split()

        context = words[-(self.model.n - 1):]
        context = self.map_oov(context)

        probs = self.model.lookup(context)

        sorted_words = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        return [w for w, _ in sorted_words[:k]]