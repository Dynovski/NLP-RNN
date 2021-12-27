import re
import string


def clean_sms_data(text: str) -> str:
    """
    Cleans text from redundant elements typical for SMS Collection corpus.
    Makes text lowercase, removes text in square brackets, removes links, html tags, words containing numbers,
    newlines, non-alphanumeric characters and redundant spaces.

    :return: str
        Preprocessed text
    """
    text = re.sub(r"&gt", ">", text)
    text = re.sub(r"&lt", "<", text)
    text = re.sub(r"&amp", "&", text)

    text = re.sub(r"gt", ">", text)
    text = re.sub(r"lt", "<", text)
    text = re.sub(r"amp", "&", text)

    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'[^a-zA-Z0-9]+', ' ', text)
    text = re.sub(r'/\s\s+/g', ' ', text)
    text = text.strip()

    return text


def clean_tweets_data(text: str) -> str:
    """
    Cleans text from redundant elements typical for Disaster Tweets corpus.
    Makes text lowercase. Removes emojis, twitter annotations, html tags, links, newlines, words with numbers,
    punctuation, redundant spaces

    :return: str
        Preprocessed text
    """
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE
    )

    text = re.sub(r"&gt", ">", text)
    text = re.sub(r"&lt", "<", text)
    text = re.sub(r"&amp", "&", text)

    text = re.sub(r"gt", ">", text)
    text = re.sub(r"lt", "<", text)
    text = re.sub(r"amp", "&", text)

    text = text.lower()
    text = re.sub(emoji_pattern, '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'[^a-zA-Z0-9]+', ' ', text)
    text = re.sub(r'/\s\s+/g', ' ', text)
    text = text.strip()

    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)


def clean_news_data(text: str) -> str:
    """
    Cleans text from redundant elements typical for AG News corpus.
    Makes text lowercase. Removes links, newlines, punctuation, numbers and redundant spaces

    :return: str
        Preprocessed text
    """
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'/\s\s+/g', ' ', text)
    text = text.strip()

    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

# TODO: spelling correction
# from spellchecker import SpellChecker
#
# spell = SpellChecker()
# def correct_spellings(text):
#     corrected_text = []
#     misspelled_words = spell.unknown(text.split())
#     for word in text.split():
#         if word in misspelled_words:
#             corrected_text.append(spell.correction(word))
#         else:
#             corrected_text.append(word)
#     return " ".join(corrected_text)
