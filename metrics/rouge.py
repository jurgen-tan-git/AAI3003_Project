import os

import pandas as pd
from rouge_metric import PyRouge
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer

ARTICLES_DIR = "./articles"


def summarize(text, language="english", sentences_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join([str(sentence) for sentence in summary])


def evaluate_summary(summary, reference):
    rouge = PyRouge(
        rouge_n=(1, 2, 4),
        rouge_l=True,
        rouge_w=True,
        rouge_w_weight=1.2,
        rouge_s=True,
        rouge_su=True,
        skip_gap=4,
    )
    scores = rouge.evaluate([summary], [[reference]])
    return scores


def main():
    summaries = {}
    for root, _, files in os.walk(ARTICLES_DIR):
        for file in files:
            if file.endswith(".txt") and file not in summaries:
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    article = f.read()
                    summary = summarize(article)
                    summaries[file] = {"text": article, "summary": summary}

    rouge_scores = {
        "rouge-1-r": [],
        "rouge-1-p": [],
        "rouge-1-f": [],
        "rouge-2-r": [],
        "rouge-2-p": [],
        "rouge-2-f": [],
        "rouge-4-r": [],
        "rouge-4-p": [],
        "rouge-4-f": [],
        "rouge-l-r": [],
        "rouge-l-p": [],
        "rouge-l-f": [],
        "rouge-w-1.2-r": [],
        "rouge-w-1.2-p": [],
        "rouge-w-1.2-f": [],
        "rouge-s4-r": [],
        "rouge-s4-p": [],
        "rouge-s4-f": [],
        "rouge-su4-r": [],
        "rouge-su4-p": [],
        "rouge-su4-f": [],
    }

    for file, data in summaries.items():
        scores = evaluate_summary(data["summary"], data["text"])
        for key, res in scores.items():
            for k, v in res.items():
                if f"{key}-{k}" in rouge_scores:
                    rouge_scores[f"{key}-{k}"].append(v)

    df = pd.DataFrame.from_dict(rouge_scores)
    print(df.describe())
    print(df.head())


if __name__ == "__main__":
    main()
