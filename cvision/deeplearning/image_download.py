"""
Form a subset of the Flickr Style data, download images to dirname, and write
Caffe ImagesDataLayer training file.
"""
import os
import urllib
import argparse
import numpy as np
import pandas as pd
import multiprocessing
import traceback
import sys

# Flickr returns a special image if the request is unavailable.
# MISSING_IMAGE_SHA1 = '6a92790b1c2a301c6e7ddef645dca1f53ea97ac2'

example_dirname = os.path.abspath(os.path.dirname(__file__))
dataset_dirname = os.path.abspath(os.path.join(example_dirname, '../../dataset'))
image_dirname = os.path.join(dataset_dirname, 'image/hannover')


def download_image(args_tuple):
    "For use with multiprocessing map. Returns filename on fail."
    try:
        url, filename = args_tuple
        #if filename.split(".")[-1] == "gif":  #could not process gif files
        #    return False
        if not os.path.exists(filename):
            urllib.urlretrieve(url, filename)
        # if Image.open(filename) and imghdr.what(filename) != "gif":
        #     return True
        return True
    except KeyboardInterrupt:
        raise Exception()  # multiprocessing doesn't catch keyboard exceptions
    except:
        traceback.print_exc(file=sys.stdout)
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download a subset of image with specific color!')
    parser.add_argument(
        '-s', '--seed', type=int, default=0,
        help="random seed")
    parser.add_argument(
        '-i', '--images', type=int, default=-1,
        help="number of images to use (-1 for all [default])",
    )
    parser.add_argument(
        '-w', '--workers', type=int, default=-1,
        help="num workers used to download images. -x uses (all - x) cores [-1 default]."
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    # Read data, shuffle order, and subsample.
    csv_filename = os.path.join(example_dirname, 'hannover_308.csv')
    df = pd.read_csv(csv_filename) #, index_col=0, compression='gzip')
    df = df.iloc[np.random.permutation(df.shape[0])]
    if args.images > 0 and args.images < df.shape[0]:
        df = df.iloc[:args.images]

    # Make directory for images and get local filenames.
    if image_dirname is None:
        image_dirname = os.path.join(dataset_dirname, 'image/hannover')
   # image_dirname = os.path.join(training_dirname, 'images')
    if not os.path.exists(image_dirname):
        os.makedirs(image_dirname)

   #create directories for each category
    cats = df['cat_id'].unique()
    for c in cats:
        p = os.path.join(image_dirname, str(c))
        print p
        if not os.path.exists(p):
            os.makedirs(p)

    df['image_filename'] = [
        os.path.join(image_dirname, str(c) + "/" + str(k) + "." + v.split(".")[-1]) for c, k, v in zip(df['cat_id'], df['auction_id'], df['image_url'])
    ]

    # a = [os.path.join(image_dirname, str(c) + "/" + str(k)  + "." + v.split(".")[-1]) for c, k, v in zip(df['cat_id'], df['auction_id'], df['image_url'])]
    # print a
    # Download images.
    num_workers = args.workers
    if num_workers <= 0:
        num_workers = multiprocessing.cpu_count() + num_workers
    print('Downloading {} images with {} workers...'.format(
        df.shape[0], num_workers))
    pool = multiprocessing.Pool(processes=num_workers)
    map_args = zip(df['image_url'], df['image_filename'])
    results = pool.map(download_image, map_args)

    # Only keep rows with valid images, and write out training file lists.
    # df = df[results]
    # for split in ['train', 'test']:
    #     split_df = df[df['_split'] == split]
    #     filename = os.path.join(training_dirname, '{}.txt'.format(split))
    #     split_df[['image_filename', 'label']].to_csv(
    #         filename, sep=' ', header=None, index=None)
    print('Writing train/val for {} successfully downloaded images.'.format(
        df.shape[0]))