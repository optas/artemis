"""
A set of functions that are useful for pre-processing textual data: uniformizing the words, spelling, etc.

The MIT License (MIT)
Copyright (c) 2021 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab
"""

import re

contractions_dict = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "I had",
    "i'd've": "I would have",
    "i'll": "I will",
    "i'll've": "I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "iit will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that had",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they had",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
    "do'nt": "do not",
    "does\'nt": "does not"
}

CONTRACTION_RE = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                            flags=re.IGNORECASE | re.DOTALL)


def expand_contractions(text, contractions=None, lower_i=True):
    """ Expand the contractions of the text (if any).
    Example: You're a good father. -> you are a good father.
    :param text: (string)
    :param contractions: (dict)
    :param lower_i: boolean, if True (I'm -> 'i am' not 'I am')
    :return: (string)

    Note:
        Side-effect: lower-casing. E.g., You're -> you are.
    """
    if contractions is None:
        contractions = contractions_dict  # Use one define in this .py

    def expand_match(contraction):
        match = contraction.group(0)
        expanded_contraction = contractions.get(match)
        if expanded_contraction is None:
            expanded_contraction = contractions.get(match.lower())
        if lower_i:
            expanded_contraction = expanded_contraction.lower()
        return expanded_contraction

    expanded_text = CONTRACTION_RE.sub(expand_match, text)
    return expanded_text


QUOTES_RE_STR = r"""(?:['|"][\w]+['|"])"""    # Words encapsulated in apostrophes.
QUOTES_RE = re.compile(r"(%s)" % QUOTES_RE_STR, flags=re.VERBOSE | re.IGNORECASE | re.UNICODE)


def unquote_words(s):
    """ 'king' - > king, "queen" -> queen """
    iterator = QUOTES_RE.finditer(s)
    new_sentence = list(s)
    for match in iterator:
        start, end = match.span()
        new_sentence[start] = ' '
        new_sentence[end-1] = ' '
    new_sentence = "".join(new_sentence)
    return new_sentence


def manual_sentence_spelling(x, spelling_dictionary):
    """
    Applies spelling on an entire string, if x is a key of the spelling_dictionary.
    :param x: (string) sentence to potentially be corrected
    :param spelling_dictionary: correction map
    :return: the sentence corrected
    """
    if x in spelling_dictionary:
        return spelling_dictionary[x]
    else:
        return x


def manual_tokenized_sentence_spelling(tokens, spelling_dictionary):
    """
    :param tokens: (list of tokens) to potentially be corrected
    :param spelling_dictionary: correction map
    :return: a list of corrected tokens
    """
    new_tokens = []
    for token in tokens:
        if token in spelling_dictionary:
            res = spelling_dictionary[token]
            if type(res) == list:
                new_tokens.extend(res)
            else:
                new_tokens.append(res)
        else:
            new_tokens.append(token)
    return new_tokens


# noinspection PyInterpreter
if __name__ == "__main__":
    import pandas as pd
    text = pd.DataFrame({'data': ["I'm a 'good' MAN", "You can't be likee this."]})
    print("Original Text:")
    print(text.data)

    manual_speller = {'You can\'t be likee this.': 'You can\'t be like this.'}
    text.data = text.data.apply(lambda x: manual_sentence_spelling(x, manual_speller))
    text.data = text.data.apply(lambda x: x.lower())
    text.data = text.data.apply(unquote_words)
    text.data = text.data.apply(expand_contractions)
    print("Corrected Text:")
    print(text.data)