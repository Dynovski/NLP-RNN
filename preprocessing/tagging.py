from typing import List
from nltk.data import load
from nltk import pos_tag


class Tagger:
    def __init__(self, all_tags: List[str]):
        self.all_tags: List[str] = all_tags

    def tag_tokens(self, tokens: List[str]):
        raise NotImplementedError('subclasses must override map_sentence()!')


class FullTagger(Tagger):
    def __init__(self):
        tag_list = [self._map_tag(tag) for tag in load('help/tagsets/upenn_tagset.pickle').keys()]
        super(FullTagger, self).__init__(tag_list)

    def tag_tokens(self, tokens: List[str]):
        pos_tags = pos_tag(tokens)
        return [(token, self._map_tag(tag)) for (token, tag) in pos_tags]

    @classmethod
    def _map_tag(cls, tag: str):
        if tag in ['VBN', 'VBZ', 'VBG', 'VBP', 'VBD', 'MD', 'NN',
                   'NNPS', 'NNP', 'NNS', 'JJS', 'JJR', 'JJ', 'CD',
                   'IN', 'PDT', 'CC', 'EX', 'POS', 'RP', 'FW', 'DT',
                   'UH', 'TO', 'PRP', 'PRP$', '$', 'WP$', 'WDT', 'WRB']:
            return tag
        elif tag in ['RB', 'RBR', 'RBS']:
            return 'RB'
        elif tag == '-':
            return 'EMPTY'
        else:
            return 'OTHER'


class UniversalTagger(Tagger):
    def __init__(self):
        tag_list = [
            'ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN',
            'NUM', 'PRT', 'PRON', 'VERB', '.', 'X', 'EMPTY'
        ]
        super(UniversalTagger, self).__init__(tag_list)

    def tag_tokens(self, tokens: List[str]):
        pos_tags = pos_tag(tokens, tagset='universal')
        return [(token, self._map_tag(tag)) for (token, tag) in pos_tags]

    @classmethod
    def _map_tag(cls, tag: str):
        if tag in ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN',
                   'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']:
            return tag
        elif tag == '-':
            return 'EMPTY'
        else:
            raise NotImplementedError('Unexpected tag!')
