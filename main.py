import numpy as np

from typing import Optional, List, Tuple

import config as cfg

from preprocessing.data_preprocessing import DataPreprocessor, SmsDataPreprocessor, TweetDataPreprocessor, NewsDataPreprocessor
from dataprocessing.datasets import create_double_split_dataset, create_dataset, create_single_split_dataset
from dataprocessing.dataloaders import create_data_loader
from analyzing.analyzer import Analyzer
from analyzing.visualization import WordsVisualizer
from inference import Trainer, Tester


def analyze():
    # Create all data preprocessors
    preprocessors_list: List[List[DataPreprocessor]] = [
        [SmsDataPreprocessor()],
        [TweetDataPreprocessor()],
        [
            NewsDataPreprocessor('data/news_train.csv'),
            NewsDataPreprocessor('data/news_test.csv')
        ]
    ]

    # Run them all
    for item in preprocessors_list:
        for preprocessor in item:
            preprocessor.run()

    # Analyze datasets
    analyzers: List[Analyzer] = []
    for item in preprocessors_list:
        analyzers.append(Analyzer(item[0].data))

    distribution_filenames = ['smsClassDistribution', 'tweetsClassDistribution', 'newsClassDistribution']
    for i in range(len(analyzers)):
        analyzers[i].analyze_distribution('class', f'figures/{distribution_filenames[i]}')

    # Create word clouds
    wc_filenames = ['smsWordCloud', 'tweetWordCloud', 'newsWordCloud']
    data_attributes = ['message', 'message', 'text']
    values_to_visualize = [
        ['ham', 'spam'],
        [0, 1],
        [1, 2, 3, 4]
    ]
    for i, item in enumerate(preprocessors_list):
        visualizer: WordsVisualizer = WordsVisualizer(item[0].data)
        for j in range(len(values_to_visualize[i])):
            visualizer.create_wordcloud(
                f'figures/{wc_filenames[i]}_{values_to_visualize[i][j]}',
                'class',
                preprocessors_list[i][0].MAIN_DATA_COLUMN,
                values_to_visualize[i][j]
            )


def main(path: str, data_labels: List[str]):
    # Create and run data preprocessor
    preprocessors: Optional[List[DataPreprocessor]] = None

    if cfg.TASK_TYPE == cfg.TaskType.SMS:
        preprocessors: List[SmsDataPreprocessor] = [SmsDataPreprocessor()]
    elif cfg.TASK_TYPE == cfg.TaskType.TWEET:
        preprocessors: List[TweetDataPreprocessor] = [TweetDataPreprocessor()]
    elif cfg.TASK_TYPE == cfg.TaskType.NEWS:
        preprocessors: List[NewsDataPreprocessor] = [NewsDataPreprocessor('data/news_train.csv'),
                                                     NewsDataPreprocessor('data/news_test.csv')]

    [preprocessor.run() for preprocessor in preprocessors]

    # Get training and test data
    training_data: np.ndarray = preprocessors[0].tokenize()
    test_data: Optional[np.ndarray] = None

    if len(preprocessors) > 1:
        test_data: np.ndarray = preprocessors[1].tokenize()

    # Create datasets
    datasets = None
    if len(preprocessors) == 1:
        datasets = create_double_split_dataset(training_data, preprocessors[0].target_labels, cfg.TRAIN_DATA_RATIO)
    elif len(preprocessors) == 2:
        train_and_val_datasets = create_single_split_dataset(
            training_data,
            preprocessors[0].target_labels,
            cfg.TRAIN_DATA_RATIO
        )
        test_dataset = create_dataset(test_data, preprocessors[1].target_labels)
        datasets = train_and_val_datasets + (test_dataset,)

    # Create data loaders
    train_dl, val_dl, test_dl = [create_data_loader(dataset) for dataset in datasets]

    # Create embedding matrices
    embedding_matrix: np.ndarray = preprocessors[0].make_embedding_matrix(
        'data/glove.6B.100d.txt',
        cfg.EMBEDDING_VECTOR_SIZE
    )

    accuracy: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    for _ in range(5):
        Trainer(
            embedding_matrix,
            train_dl,
            val_dl,
            preprocessors[0].tokenizer,
            training_data.shape[1]
        ).run()
        results = Tester(
            embedding_matrix,
            datasets[2][:][0],
            datasets[2][:][1],
            preprocessors[0].tokenizer,
            training_data.shape[1]
        ).run(path, data_labels)
        accuracy += results[0]
        f1 += results[1]
        precision += results[2]
        recall += results[3]
    return accuracy / 5, f1 / 5, precision / 5, recall / 5


if __name__ == '__main__':
    # analyze()
    results = main('figures/smsConfusionMatrix.png', ['Ham', 'Spam'])
    print("\nAverage accuracy: {:.3f}%".format(results[0] * 100))
    print("Average F1-score: {:.3f}%".format(results[1] * 100))
    print("Average precision: {:.3f}%".format(results[2] * 100))
    print("Average recall: {:.3f}%".format(results[3] * 100))
