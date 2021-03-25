"""
Mostly some constants & very simple function to encode/handle the emotion attributes of ArtEmis.

The MIT License (MIT)
Originally created at 02/11/20, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""


ARTEMIS_EMOTIONS = ['amusement', 'awe', 'contentment', 'excitement',
                    'anger', 'disgust',  'fear', 'sadness', 'something else']

EMOTION_TO_IDX = {e: i for i, e in enumerate(ARTEMIS_EMOTIONS)}


IDX_TO_EMOTION = {EMOTION_TO_IDX[e]: e for e in EMOTION_TO_IDX}


POS_NEG_ELSE = {'amusement': 0, 'awe': 0, 'contentment': 0, 'excitement': 0,
                'anger': 1, 'disgust': 1,  'fear': 1, 'sadness': 1,
                'something else': 2}


COLORS = {'amusement': '#EE82EE',
          'awe': '#FFFF00',
          'contentment': '#87CEEB',
          'excitement': '#DC143C',
          'anger': '#000080',
          'disgust': '#F0E68C',
          'fear': '#C0C0C0',
          'sadness': '#696969',
          'something else': '#228B22'}


LARGER_EMOTION_VOCAB = {('bored', 'boring', 'apathy', 'boredom', 'indifferent', 'dull', 'uninteresting', 'uninterested'),
                        ('shock', 'shocked'),
                        ('confused', 'confusion', 'confuses', 'puzzled', 'puzzling',
                         'perplexed', 'perplexing', 'confusing', 'odd', 'weird'),
                        ('surprised',),
                        ('anticipation',),
                        ('empowerment',),
                        ('hope', 'hopeful', 'optimistic'),
                        ('neutral',),
                        ('rage',),
                        ('happy', 'happiness'),
                        ('grief',),
                        ('shame',),
                        ('resent',),
                        ('creepy',),
                        ('disappointment',),
                        ('depressing', 'depressed'),
                        ('bothered', 'disturbed', 'bothersome'),
                        ('overwhelmed',),
                        ('anxiety', 'anxious'),
                        ('thrilled',),
                        ('surprised', 'surprising'),
                        ('uncomfortable',),
                        ('curious', 'curiosity', 'wonder', 'intrigued', 'interested', 'interests', 'interesting', 'intriguing'),
                        ('alerted', 'alert'),
                        ('insult', 'insulted'),
                        ('shy',),
                        ('nostalgia', 'nostalgic'),
                        ('exhilarating', 'exhilarated')}


def positive_negative_else(emotion):
    """ Map a feeling string (e.g. 'awe') to an integer indicating if it is a positive, negative, or else.
    :param emotion: (string)
    :return: int
    """
    return POS_NEG_ELSE[emotion]


def emotion_to_int(emotion):
    """ Map a feeling string (e.g. 'awe') to a unique integer.
    :param emotion: (string)
    :return: int
    """
    return EMOTION_TO_IDX[emotion]