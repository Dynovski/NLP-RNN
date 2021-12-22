import pandas as pd
import numpy as np

from typing import List, Dict
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from exceptions import IncorrectNumberOfAttributesException
from preprocessing.utils import clean_sms_data


class DataPreprocessor:
    """
    Class having utilities to preprocess provided dataset. Dataset is read and being processed as pandas'
    data frame.

    ---
    Attributes:
        data: pandas.DataFrame
            Data frame containing all data
        stopwords: List[str]
            Stopwords for english for preprocessing
        stemmer: nltk.stem.SnowballStemer
            Stemmer for english used for preprocessing
        tokenizer: keras.preprocessing.text.Tokenizer
            Tokenizer used to process text data to indices
        LABEL_COLUMN: str
            Name of label column
        MAIN_DATA_COLUMN: str
            Name of column containing data to use for training
        ENCODED_TARGETS_COLUMN: str
            Name of column containing labels encoded to integers
    """
    def __init__(
            self,
            path: str,
            encoding: str,
            attribute_names: List[str],
            data_column: str,
            label_column: str
    ):
        """
        :param path: str
            Path to the dataset file stored locally
        :param encoding: str
            List of new attribute names for consecutive columns in data frame
        :param attribute_names: List[str]
            List of new attribute names for consecutive columns in data frame
        :param data_column: str
            Name of the column that contains data
        :param label_column: str
            Name of the column that contains data labels
        """
        self.data: pd.DataFrame = pd.read_csv(path, delimiter=',', encoding=encoding)

        # Remove empty columns
        self.data.dropna(axis=1, how='any', inplace=True)
        # Remove empty words
        self.data.dropna(how='any', inplace=True)

        # Check if number of attributes is correct
        if self.data.shape[1] != len(attribute_names):
            raise IncorrectNumberOfAttributesException(self.data.shape[1], len(attribute_names))

        self.data.columns = attribute_names

        self.stopwords: List[str] = stopwords.words('english')
        self.stemmer: SnowballStemmer = SnowballStemmer('english')
        self.tokenizer: Tokenizer = Tokenizer()

        self.MAIN_DATA_COLUMN = 'data_after_preprocessing'
        self.ENCODED_TARGETS_COLUMN = 'encoded_targets'

        self.data[self.MAIN_DATA_COLUMN] = self.data[data_column]

        assert label_column in attribute_names
        self.LABEL_COLUMN = label_column

    def _remove_stopwords(self, text: str) -> str:
        text = ' '.join(word for word in text.split(' ') if word not in self.stopwords)
        return text

    def _stem_text(self, text: str) -> str:
        text = ' '.join(self.stemmer.stem(word) for word in text.split(' '))
        return text

    def dataset_specific_preprocessing(self) -> None:
        """
        Preprocessing specific for the dataset

        :return: None
        """
        raise NotImplementedError('subclasses must override dataset_specific_preprocessing(attribute: str)!')

    def remove_stopwords(self) -> None:
        # Remove stopwords
        self.data[self.MAIN_DATA_COLUMN] = self.data[self.MAIN_DATA_COLUMN].apply(self._remove_stopwords)

    def stem(self) -> None:
        """
        Applies stemming so data

        :return: None
        """
        assert self.MAIN_DATA_COLUMN in self.data.columns

        self.data[self.MAIN_DATA_COLUMN] = self.data[self.MAIN_DATA_COLUMN].apply(self._stem_text)

    def encode_targets(self) -> None:
        """
        Encodes target labels from categorical to numeric values

        :return: None
        """
        assert self.LABEL_COLUMN in self.data.columns

        encoder: LabelEncoder = LabelEncoder()
        encoder.fit(self.data[self.LABEL_COLUMN])
        self.data[self.ENCODED_TARGETS_COLUMN] = encoder.transform(self.data[self.LABEL_COLUMN])

    def run(self) -> None:
        """
        Call this function to preprocess the data

        :return: None
        """
        raise NotImplementedError('subclasses must override run(data_attr: str, label_attr: str)!')

    @property
    def training_data(self) -> List[str]:
        return self.data[self.MAIN_DATA_COLUMN].tolist()

    @property
    def target_labels(self) -> np.ndarray:
        assert self.ENCODED_TARGETS_COLUMN in self.data.columns

        return self.data[self.ENCODED_TARGETS_COLUMN].to_numpy(dtype=np.float32)

    def tokenize(self) -> np.ndarray:
        """
        Tokenize preprocessed data, convert it to numerical values and pad it with zeroes at the end

        :return: numpy.ndarray
             Array of numerical values where each row corresponds to the same row of preprocessed data
        """
        self.tokenizer.fit_on_texts(self.training_data)

        longest_sequence: str = max(self.training_data, key=lambda t: len(word_tokenize(t)))
        max_len: int = len(word_tokenize(longest_sequence))

        padded_sequences: np.ndarray = pad_sequences(
            self.tokenizer.texts_to_sequences(self.training_data),
            max_len,
            padding='post'
        )

        return padded_sequences

    def make_embedding_matrix(self, path: str, dim: int, *, stem: bool = False) -> np.ndarray:
        """
        Creates glove embedding matrix. Must be called after tokenize()

        :param path: str
            Path to locally stored file containing pretrained glove model
        :param dim: int
            Dimension of glove vector
        :param stem: bool
            Flag specifying whether to create embedding matrix for stems only

        :return: numpy.ndarray
            Matrix containing vector representation for each consecutive word in vocabulary
        """
        word_to_vector_d: Dict[str, np.ndarray] = dict()

        with open(path, encoding='utf8') as f:
            for line in tqdm(f, "Reading GloVe"):
                elements = line.split()
                word = self.stemmer.stem(elements[0]) if stem else elements[0]
                vector = np.asarray(elements[1:], dtype='float32')
                word_to_vector_d[word] = vector

        embedding_matrix = np.zeros((len(self.tokenizer.word_index) + 1, dim))

        count: int = 0
        for word, index in self.tokenizer.word_index.items():
            vector = word_to_vector_d.get(word)
            if vector is not None:
                count += 1
                embedding_matrix[index] = vector

        print(f'There are embeddings for {count} words of {len(self.tokenizer.word_index) + 1} words in vocabulary')

        return embedding_matrix


class SmsDataPreprocessor(DataPreprocessor):
    def __init__(self):
        super(SmsDataPreprocessor, self).__init__(
            path='data/SMSDataset.csv',
            encoding='latin-1',
            attribute_names=['class', 'message'],
            data_column='message',
            label_column='class'
        )

    def dataset_specific_preprocessing(self) -> None:
        self.data[self.MAIN_DATA_COLUMN] = self.data[self.MAIN_DATA_COLUMN].apply(clean_sms_data)

    def run(self) -> None:
        self.dataset_specific_preprocessing()
        self.stem()
        self.encode_targets()
