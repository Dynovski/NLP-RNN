import json


class IndexMapper:
    def __init__(self, tokenizer):
        self.dictionary = json.loads(tokenizer.get_config()['index_word'])

    def index_to_word(self, index):
        if index == 0:
            return ''
        return self.dictionary[str(index)]

    def indices_to_words(self, indices):
        word_list = []
        for index in indices:
            word_list.append(self.index_to_word(index))
        return word_list
