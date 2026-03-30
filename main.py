import argparse
import os
from dotenv import load_dotenv

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor


# def run_dataprep(normalizer):
#     text = normalizer.load(os.getenv("TRAIN_RAW_DIR"))
#     text = normalizer.normalize(text)

#     sentences = normalizer.sentence_tokenize(text)
#     tokenized = [normalizer.word_tokenize(s) for s in sentences]

#     normalizer.save(tokenized, os.getenv("TRAIN_TOKENS"))


def run_dataprep(normalizer):
    # 1. Load raw text
    text = normalizer.load(os.getenv("TRAIN_RAW_DIR"))

    # 2. Strip Gutenberg ONCE
    text = normalizer.strip_gutenberg(text)

    # 3. Sentence tokenize BEFORE normalization
    sentences = normalizer.sentence_tokenize(text)

    # 4. Normalize each sentence
    normalized_sentences = [
        normalizer.normalize(sentence)
        for sentence in sentences
        if sentence.strip()
    ]

    # 5. Word tokenize
    tokenized = [
        normalizer.word_tokenize(sentence)
        for sentence in normalized_sentences
    ]

    # 6. Save
    normalizer.save(tokenized, os.getenv("TRAIN_TOKENS"))






def run_model(model):
    model.build_vocab(
        os.getenv("TRAIN_TOKENS"),
        int(os.getenv("UNK_THRESHOLD")),
    )
    model.build_counts_and_probabilities(os.getenv("TRAIN_TOKENS"))
    model.save_model(os.getenv("MODEL"))
    model.save_vocab(os.getenv("VOCAB"))


def run_inference(model, normalizer):
    model.load(os.getenv("MODEL"), os.getenv("VOCAB"))
    predictor = Predictor(model, normalizer)

    while True:
        text = input("> ")
        if text == "quit":
            print("Goodbye.")
            break

        print(predictor.predict_next(text, int(os.getenv("TOP_K"))))


def main():
    load_dotenv("config/.env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True)
    args = parser.parse_args()

    normalizer = Normalizer()
    model = NGramModel(int(os.getenv("NGRAM_ORDER")))

    if args.step == "dataprep":
        run_dataprep(normalizer)

    elif args.step == "model":
        run_model(model)

    elif args.step == "inference":
        run_inference(model, normalizer)

    elif args.step == "all":
        run_dataprep(normalizer)
        run_model(model)
        run_inference(model, normalizer)


if __name__ == "__main__":
    main()