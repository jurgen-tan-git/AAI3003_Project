"""Summarise articles using LSA summarizer from sumy library.
"""

import os

from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer


def summarize(text: str, language: str = "english", sentences_count: int = 5) -> str:
    """Return a summary of the given text using LSA summarization technique.

    :param text: Text to summarize.
    :type text: str
    :param language: Language for the tokenizer, defaults to "english"
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


def main():
    """The main function to summarize articles using LSA summarizer."""
    categories = os.listdir("./articles")
    for category in categories:
        for file in os.listdir(f"./articles/{category}"):
            with open(f"./articles/{category}/{file}", "r", encoding="utf-8") as f:
                article = f.read()
                print("Original Word Count: {}".format(len(article)))

            summary = summarize(article)

            # Make folder to store summaries
            if not os.path.exists(f"summaries/{category}"):
                os.makedirs(f"summaries/{category}")

            with open(f"summaries/{category}/{file}", "w", encoding="utf-8") as f:
                f.write(summary)
                print("Summary Word Count: {}".format(len(summary)))


if __name__ == "__main__":
    main()
