import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.plotting import plot_confusion_matrix
from typing import List

import config
from preprocessing.data_preprocessing import SmsDataPreprocessor


class Analyzer:
    def __init__(self, data: 'pandas.DataFrame'):
        self.data = data

    def analyze_distribution(self, attribute: str, path: str = '') -> 'pandas.DataFrame':
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

        distribution: 'pandas.DataFrame' = self.data.groupby(attribute)[attribute].agg('count')

        if path:
            plt.figure(figsize=(12, 8))
            plot = sns.countplot(self.data[attribute])
            plot.set_title("Count plot of classes")
            plot.set_xlabel("Classes")
            plot.set_ylabel("Number of data points")
            fig = plot.get_figure()
            fig.savefig(path)

        return distribution

    @staticmethod
    def plot_confusion_matrix(data, labels: 'List[str]'):
        plt.figure()
        plot_confusion_matrix(data, figsize=(16, 12), hide_ticks=True, cmap=plt.cm.Blues)
        plt.xticks(range(len(labels)), labels, fontsize=12)
        plt.yticks(range(len(labels)), labels, fontsize=12)
        plt.savefig(f'figures/{config.CM_NAME}')

    @staticmethod
    def plot_losses(train_loss: List[float], val_loss: List[float], steps: List[int]):
        plt.figure()
        plt.plot(steps, train_loss, 'r', label='Training loss')
        plt.plot(steps, val_loss , 'b', label='Validation loss')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(config.LOSS_PLOT_NAME)
        plt.savefig(f'figures/{config.LOSS_PLOT_NAME}')


def analyze() -> None:
    preprocessor = SmsDataPreprocessor()
    preprocessor.run()
    analyzer = Analyzer(preprocessor.data)
    print(analyzer.analyze_distribution('class', 'figures/SmsClassDistribution.png'))


if __name__ == '__main__':
    analyze()
