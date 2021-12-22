import re
import string


def clean_sms_data(text: str) -> str:
    """
    Cleans text from redundant elements typical for SMS Collection corpus.
    Makes text lowercase, removes text in square brackets, removes links, removes punctuation
    and removes words containing numbers.

    :return: str
        Preprocessed text
    """
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    text = re.sub(r'/\s\s+/g', ' ', text)
    text = text.strip()

    return text


def clean_tweets_data(text: str) -> str:
    """
    Cleans text from redundant elements typical for Disaster Tweets corpus.

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
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[#@]\w+', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'/\s\s+/g', ' ', text)
    text = text.strip()

    return text


def clean_news_data(text: str) -> str:
    """
    Cleans text from redundant elements typical for AG News corpus.

    :return: str
        Preprocessed text
    """
    # TODO: make regex rules to clean news
    pass
