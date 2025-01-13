#!/usr/bin/env python
"""
python script to run crop_segmentation with multiple processes for each year
usage: crop_segmentation_multiprocessing [number of parallel processes]
"""

import sys
import os
import logging
from multiprocessing import Pool
import shutil
import warnings
warnings.filterwarnings("ignore")

def mute():
    sys.stdout = open(os.devnull, 'w')

if __name__ == '__main__':
 
    import importlib
    crop_segmentation = importlib.import_module("01_crop_segmentation")
    
    logging.basicConfig(level=logging.ERROR)

    run_in_parallel = int(sys.argv[1])
    if run_in_parallel == None:
        logging.error('How many processes in parallel? -> usage: crop_segmentation_multiprocessing [number of parallel processes]')
        sys.exit()

    # create output structure
    output_dir = os.path.join("../data", "000_Raw_Images",'cropped_tables')
    if os.path.exists(output_dir):
        print('cropped tables already existing. Check results first and delete them if you want to proceed')
        sys.exit()
    os.makedirs(output_dir)

    # prepare input of all processes
    years = os.listdir('../data/000_Raw_Images/pictures_all')
    arg_list = [[] for _ in range(len(years))]
    for idx,year in enumerate(years):
        tmp = year
        arg_list[idx] = (tmp,)

    p = Pool(run_in_parallel,initializer=mute)
    p.starmap(crop_segmentation.main, arg_list)

    logging.info('All cropping processes done ::: ')