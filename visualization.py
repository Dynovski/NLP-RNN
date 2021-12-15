from wordcloud import WordCloud

from matplotlib import pyplot as plt


class WordsVisualizer:
    def __init__(self, data: 'pd.DataFrame'):
        self.wordcloud = WordCloud(
            background_color='white',
            max_words=200
        )
        self.data: 'pd.DataFrame' = data

    def create_wordcloud(self, path: str, label_attr: str, data_attr: str, value: str, show: bool = False) -> None:
        """
        Generates wordcloud and saves it in path

        :param path: str
            Path to the location to which save generated wordcloud
        :param label_attr: str
            Name of column specifying labels
        :param data_attr: str
            Name of column specifying data
        :param value: str
            Value in labels column for which to create wordcloud
        :param show: bool
            Flag, if set wordcloud will be shown, if not it will be only saved to path
        :return: None
        """
        self.wordcloud.generate(' '.join(text for text in self.data.loc[self.data[label_attr] == value, data_attr]))

        plt.figure(figsize=(18, 10))
        plt.title(
            f'Top words for {value}',
            fontdict={'size': 22, 'verticalalignment': 'bottom'}
        )
        plt.imshow(self.wordcloud)
        plt.axis("off")
        plt.savefig(path)

        if show:
            plt.show()
