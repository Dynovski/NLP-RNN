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
    text = re.sub(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'/\s\s+/g', ' ', text)
    text = text.strip()

    return text


def clean_tweets_data(text: str) -> str:
    """
    Cleans text from redundant elements typical for Disaster Tweets corpus.

    :return: str
        Preprocessed text
    """
    # TODO: make regex rules to clean tweets
    pass


def clean_news_data(text: str) -> str:
    """
    Cleans text from redundant elements typical for AG News corpus.

    :return: str
        Preprocessed text
    """
    # TODO: make regex rules to clean news
    pass
