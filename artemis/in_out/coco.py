"""
COCO related I/O operations

The MIT License (MIT)
Originally created at 10/18/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import os.path as osp

def coco_image_name_to_image_file(image_name, top_img_dir, year=2014):
    if image_name.startswith('COCO_val'):
        return osp.join(top_img_dir, 'val' + str(year), image_name)
    elif image_name.startswith('COCO_train'):
        return osp.join(top_img_dir, 'train' + str(year), image_name)
    else:
        raise ValueError


def karpathize(df):
    ## Per Karpathy's tweet: restval is actually train.
    df.split[df.split == 'restval'] = 'train'


def prepare_coco_dataframe_for_training(df, top_img_dir):
    # assign file-names to each image
    df = df.assign(image_files = df.image.apply(lambda x: coco_image_name_to_image_file(x, top_img_dir)))
    # fix splits
    karpathize(df)
    return df