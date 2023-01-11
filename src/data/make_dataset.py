# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def load_data(data_paths):
    "Opens a .npz set via a path and returns it."
    images = []
    labels = []
    
    for set_path in data_paths:
        with np.load(set_path) as f:
            images_set = f['images']
            labels_set = f['labels']
            for image in images_set:
                images.append(image.astype(np.float32))
            for label in labels_set:
                labels.append(label.astype(np.float32)) 
    
    return np.stack(images), np.stack(labels)


def transform_data(data, transform=None):
    "Applies torch transformation to array"
    
    data_tensor = []
    for element in data:
        if transform != None:
            data_tensor.append(transform(element))
        else:
            data_tensor.append(torch.as_tensor(element))
    
    return torch.stack(data_tensor)

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Get all train paths
    train_paths = list(Path(input_filepath).glob("train*.npz")) 
    
    # Get test paths
    test_paths = list(Path(input_filepath).glob("test*.npz")) 

    # Define transform (mean=0 and std=1) to apply to images
    transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.,), (1.,))])

    # Import data
    trainimages, trainlabels = load_data(train_paths)
    testimages, testlabels = load_data(test_paths)

    # Transform data and create tensor
    trainimages, trainlabels = transform_data(trainimages, transform), transform_data(trainlabels, None)
    testimages, testlabels = transform_data(testimages, transform), transform_data(testlabels, None)

    # Save output
    output = {'train_data': trainimages,
                'train_labels': trainlabels,
                'test_data': testimages,
                'test_labels': testlabels}

    torch.save(output, os.path.join(output_filepath, 'train_test_processed.pt'))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()


