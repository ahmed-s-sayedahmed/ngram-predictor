import argparse
from dotenv import load_dotenv
import os

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor

def main():
    # load_dotenv()
    load_dotenv("config/.env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True)
    args = parser.parse_args()

    normalizer = Normalizer()
    model = NGramModel(int(os.getenv("NGRAM_ORDER")))
    


    if args.step == "dataprep":
        text = normalizer.load(os.getenv("TRAIN_RAW_DIR"))
        text = normalizer.strip_gutenberg(text)
        text = normalizer.normalize(text)

        sentences = normalizer.sentence_tokenize(text)
        tokenized = [normalizer.word_tokenize(s) for s in sentences]

        normalizer.save(tokenized, os.getenv("TRAIN_TOKENS"))

    elif args.step == "model":
        model.build_vocab(os.getenv("TRAIN_TOKENS"), int(os.getenv("UNK_THRESHOLD")))
        model.build_counts_and_probabilities(os.getenv("TRAIN_TOKENS"))
        model.save_model(os.getenv("MODEL"))
        model.save_vocab(os.getenv("VOCAB"))

    elif args.step == "inference":
        model.load(os.getenv("MODEL"), os.getenv("VOCAB"))
        predictor = Predictor(model, normalizer)

        while True:
            text = input("> ")
            if text == "quit":
                break

            print(predictor.predict_next(text, int(os.getenv("TOP_K"))))

if __name__ == "__main__":
    main()

