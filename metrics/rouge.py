"""This script evaluates the ROUGE scores of the summaries generated by the LSA summarization technique.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rouge_metric import PyRouge
from rouge_metric.py_rouge import RougeType
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer

ARTICLES_DIR = "./articles"


def summarize(text: str, language: str = "english", sentences_count: int = 5) -> str:
    """Summarize a given text using LSA summarization technique.

    :param text: Text to summarize.
    :type text: str
    :param language: Language for the Tokenizer, defaults to "english"
    :type language: str, optional
    :param sentences_count: Number of sentences to output, defaults to 5
    :type sentences_count: int, optional
    :return: Summarized text
    :rtype: str
    """
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join([str(sentence) for sentence in summary])


def evaluate_summary(summary: str, reference: str) -> RougeType | list[RougeType]:
    """Evaluate the ROUGE scores for a given summary and reference text.

    :param summary: Summary text.
    :type summary: str
    :param reference: Reference text.
    :type reference: str
    :return: Resulting ROUGE scores.
    :rtype: RougeType | list[RougeType]
    """
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
    """The main function to evaluate the ROUGE scores of the summaries."""
    # Set the theme for the plots
    sns.set_theme("paper", "whitegrid")
    summaries = {}

    # Generate summaries for all articles
    for root, _, files in os.walk(ARTICLES_DIR):
        for file in files:
            if file.endswith(".txt") and file not in summaries:
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    article = f.read()
                    summary = summarize(article)
                    summaries[file] = {"text": article, "summary": summary}

    # Initialize the ROUGE scores
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

    # Reorganise the scores for plotting and analysis
    for file, data in summaries.items():
        scores = evaluate_summary(data["summary"], data["text"])
        for key, res in scores.items():
            for k, v in res.items():
                if f"{key}-{k}" in rouge_scores:
                    rouge_scores[f"{key}-{k}"].append(v)

    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.max_columns", None)
    df = pd.DataFrame.from_dict(rouge_scores)
    print(df.describe())
    print(df.head())

    # Display full histograms
    df = df.melt(value_vars=rouge_scores.keys(), var_name="metric", value_name="score")
    _, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(
        df,
        x="score",
        hue="metric",
        bins=20,
        kde=False,
        common_norm=True,
        multiple="layer",
        ax=ax,
    )
    ax.set_title("ROUGE Scores Histograms")
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    plt.show()

    # Display just the F1 scores
    df_f = df[df["metric"].str.contains("-f")]
    _, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(
        df_f,
        x="score",
        hue="metric",
        bins=20,
        kde=False,
        common_norm=True,
        multiple="layer",
        ax=ax,
    )
    ax.set_title("ROUGE F1 Scores Histograms")
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    plt.show()

    # Display just the Recall scores
    df_r = df[df["metric"].str.contains("-r")]
    _, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(
        df_r,
        x="score",
        hue="metric",
        bins=20,
        kde=False,
        common_norm=True,
        multiple="layer",
        ax=ax,
    )
    ax.set_title("ROUGE Recall Scores Histograms")
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    plt.show()

    # Display just the Precision scores
    df_p = df[df["metric"].str.contains("-p")]
    _, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(
        df_p,
        x="score",
        hue="metric",
        bins=20,
        kde=False,
        common_norm=True,
        multiple="layer",
        ax=ax,
    )
    ax.set_title("ROUGE Precision Scores Histograms")
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    main()
