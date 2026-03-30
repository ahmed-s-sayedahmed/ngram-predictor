import os
import re

class Normalizer:
    def load(self, folder_path):
        texts = []
        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                    texts.append(f.read())
        return "\n".join(texts)

    def strip_gutenberg(self, text):
        start = "*** START OF THE PROJECT GUTENBERG EBOOK"
        end = "*** END OF THE PROJECT GUTENBERG EBOOK"
        if start in text and end in text:
            text = text.split(start, 1)[1]
            text = text.split(end, 1)[0]
        return text

    def lowercase(self, text):
        return text.lower()

    def remove_punctuation(self, text):
        return re.sub(r"[^\w\s]", "", text)

    def remove_numbers(self, text):
        return re.sub(r"\d+", "", text)

    def remove_whitespace(self, text):
        return re.sub(r"\s+", " ", text).strip()

    def normalize(self, text):
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_whitespace(text)
        return text

    def sentence_tokenize(self, text):
        return text.split(".")   # simple version

    def word_tokenize(self, sentence):
        return sentence.split()

    def save(self, sentences, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            for sentence in sentences:
                f.write(" ".join(sentence) + "\n")