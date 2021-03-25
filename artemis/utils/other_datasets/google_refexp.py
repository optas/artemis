"""
Minimal operations for loading google_refexp dataset.

The MIT License (MIT)
Originally created in mid 2020, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import json
import pandas as pd


def load_google_refexp_captions(train_file, val_file):
    def _load_google_refexp_json(json_file):
        with open(json_file) as fin:
            data = json.load(fin)

        ref_id_to_utterance=\
        {anno['refexp_id']: anno['raw'] for anno in data['refexps']}

        utterances = []
        images = []
        for val in data['annotations']:
            for x in val['refexp_ids']:
                utter = ref_id_to_utterance[x]
                images.append(val['image_id'])
                utterances.append(utter)

        df = pd.concat([pd.Series(images), pd.Series(utterances)], axis=1)
        return df

    df1 = _load_google_refexp_json(train_file)
    df2 = _load_google_refexp_json(val_file)
    df = pd.concat([df1, df2])
    df.reset_index(drop=True, inplace=True)
    df = df.rename(columns={0:'image_id', 1:'utterance'})
    return df