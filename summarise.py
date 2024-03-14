from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import os

def summarize(text, language="english", sentences_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return ' '.join([str(sentence) for sentence in summary])

if __name__ == "__main__":
    categories = os.listdir('./articles')
    for category in categories:
        for file in os.listdir(f'./articles/{category}'):
            with open(f'./articles/{category}/{file}', 'r', encoding='utf-8') as f:
                article = f.read()
                print("Original Word Count: {}".format(len(article)))

            summary = summarize(article)

            # Make folder to store summaries
            if not os.path.exists(f'summaries/{category}'):
                os.makedirs(f'summaries/{category}')

            with open(f'summaries/{category}/{file}', 'w', encoding='utf-8') as f:
                f.write(summary)
                print("Summary Word Count: {}".format(len(summary)))