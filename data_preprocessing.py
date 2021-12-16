import pandas as pd
import numpy as np

from typing import List, Tuple, Dict
from plotly import graph_objs as go
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from exceptions import IncorrectNumberOfAttributesException
from preprocessing.utils import clean_sms_data


class DataPreProcessor:
    """
    Class having utilities to preprocess provided dataset. Dataset is read and being processed as pandas'
    data frame.

    ---
    Attributes:
        path: str
            Path to the dataset file stored locally
        encoding: str
            Encoding used to read dataset file
        attribute_names: List[str]
            List of new attribute names for consecutive columns in data frame
        calculate_len: Tuple[bool, str]
            Argument specifying if length of each row in data frame should be calculated. First element is
            a flag, second element specifies attribute name for which length should be calculated
        data: pandas.DataFrame
            Data frame containing all data
        stopwords: List[str]
            Stopwords for english for preprocessing
        stemmer: nltk.stem.SnowballStemer
            Stemmer for english used for preprocessing
        tokenizer: keras.preprocessing.text.Tokenizer
            Tokenizer used to process text data to indices
    """
    def __init__(
            self,
            path: str,
            encoding: str,
            attribute_names: List[str],
            calculate_len: Tuple[bool, str] = (False, '')
    ):
        self.data: pd.DataFrame = pd.read_csv(path, encoding=encoding)

        # Remove empty columns
        self.data.dropna(axis=1, how='any', inplace=True)
        # Remove empty words
        self.data.dropna(how='any', inplace=True)

        # Check if number of attributes is correct
        if self.data.shape[1] != len(attribute_names):
            raise IncorrectNumberOfAttributesException(self.data.shape[1], len(attribute_names))
        self.data.columns = attribute_names

        if calculate_len[0]:
            assert calculate_len[1] in self.data.columns
            self.data['data_len'] = self.data[calculate_len[1]].apply(lambda x: len(x.split(' ')))

        self.stopwords: List[str] = stopwords.words('english')
        self.stemmer: SnowballStemmer = SnowballStemmer('english')
        self.tokenizer: Tokenizer = Tokenizer()

    def max_element_length(self) -> int:
        """
        If length of data is computed, returns length of the longest data

        :return: int
            Length of the longest data
        """
        assert 'data_len' in self.data.columns
        return max(self.data['data_len'])

    def analyze_distribution(self, attribute: str, path: str = '') -> pd.DataFrame:
        """
        Checks how many instances of given class of selected attributes is there in data.
        By specifying path, histogram is created and saved at provided location.

        :param attribute: str
            Name of the attribute for which to check distribution
        :param path: str
            Path to the file to save distribution plot
        :return: pandas.DataFrame
            data frame containing class names and number of instances for given attribute
        """
        assert attribute in self.data.columns

        distribution: pd.DataFrame = self.data.groupby(attribute)[attribute].agg('count')

        if path:
            fig = go.Figure()
            for column in distribution.columns:
                fig.add_trace(
                    go.Bar(
                        x=[column],
                        y=[distribution[column][0]],
                        name=column,
                        text=[distribution[column][0]],
                        textposition='auto'
                    )
                )
            fig.update_layout(
                title='<span style="font-size:32px; font-family:Times New Roman">Dataset distribution by target</span>'
            )
            fig.write_image(path)

        return distribution

    def _remove_stopwords(self, text: str) -> str:
        text = ' '.join(word for word in text.split(' ') if word not in self.stopwords)
        return text

    def _stem_text(self, text: str) -> str:
        text = ' '.join(self.stemmer.stem(word) for word in text.split(' '))
        return text

    def clean_data(self, attribute: str) -> None:
        """
        Cleans data specified by attribute

        :param attribute: str
            Name of column containing data
        :return: None
        """
        assert attribute in self.data.columns

        self.data['clean_data'] = self.data[attribute].apply(clean_sms_data)

        # Remove stopwords
        self.data['clean_data'] = self.data['clean_data'].apply(self._remove_stopwords)

    def stem_data(self) -> None:
        """
        Applies stemming so data

        :return: None
        """
        assert 'clean_data' in self.data.columns
        self.data['clean_data'] = self.data['clean_data'].apply(self._stem_text)

    def encode_labels(self, attribute: str) -> None:
        """
        Encodes labels from categorical to numeric values

        :param attribute: str
            Name of the attribute containing labels for data
        :return: None
        """
        encoder: LabelEncoder = LabelEncoder()
        encoder.fit(self.data[attribute])
        self.data[f'encoded_labels'] = encoder.transform(self.data[attribute])

    def get_data(self) -> pd.DataFrame:
        """
        Function to access to data stored in DataPreprocessor

        :return: pandas.Dataframe
            Data stored in DataPreProcessor object
        """
        return self.data

    def preprocess(self, data_attr: str, label_attr: str) -> pd.DataFrame:
        """
        Call this function to implicitly apply all possible processing to the data

        :param data_attr: str
            name of an attribute containing data to process
        :param label_attr: str
            name of an attribute containing labels of data
        :return: pandas.Dataframe
            Preprocessed data
        """
        self.clean_data(data_attr)
        self.stem_data()
        self.encode_labels(label_attr)

        return self.data

    @property
    def preprocessed_data(self):
        return self.data['clean_data']

    @property
    def encoded_labels(self) -> np.ndarray:
        return self.data['encoded_labels'].to_numpy(dtype=np.float32)

    def tokenize(self) -> np.ndarray:
        """
        Tokenize preprocessed data, convert it to numerical values and pad it with zeroes at the end

        :return: numpy.ndarray
             Array of numerical values where each row corresponds to the same row of preprocessed data
        """
        self.tokenizer.fit_on_texts(self.preprocessed_data)

        longest_sequence: str = max(self.preprocessed_data, key=lambda t: len(word_tokenize(t)))
        longest_sentence_len: int = len(word_tokenize(longest_sequence))

        padded_sequences: np.ndarray = pad_sequences(
            self.tokenizer.texts_to_sequences(self.preprocessed_data),
            longest_sentence_len,
            padding='post'
        )

        return padded_sequences

    def create_glove_embedding_matrix(self, path: str, embedding_dim: int, vocab_len: int) -> np.ndarray:
        """
        Creates glove embedding matrix

        :param path: str
            Path to locally stored file containing pretrained glove model
        :param embedding_dim: int
            Dimension of glove vector
        :param vocab_len: int
            Number of unique tokens
        :return: numpy.ndarray
            Matrix containing vector representation for each consecutive word in vocabulary
        """
        word_to_vector_d: Dict[str, np.ndarray] = dict()

        with open(path, encoding='utf8') as f:
            for line in tqdm(f, "Reading GloVe"):
                elements = line.split()
                word = elements[0]
                vector = np.asarray(elements[1:], dtype='float32')
                word_to_vector_d[word] = vector

        embedding_matrix = np.zeros((vocab_len, embedding_dim))

        count: int = 0
        for word, index in self.tokenizer.word_index.items():
            vector = word_to_vector_d.get(word)
            if vector is not None:
                count += 1
                embedding_matrix[index] = vector

        print(f'Glove has {count} of {len(self.tokenizer.word_index) + 1} words in vocabulary')

        return embedding_matrix
