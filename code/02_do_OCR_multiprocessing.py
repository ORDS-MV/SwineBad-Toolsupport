#!/usr/bin/env python
"""
python script to run ocr_tesseract with multiple processes for each year
usage: do_OCR_multiprocessing [number of parallel processes]
"""

import sys
import os
import logging
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")

def mute():
    sys.stdout = open(os.devnull, 'w')

if __name__ == '__main__':

    import importlib
    ocr_tesseract = importlib.import_module("02_OCR_tesseract")

    logging.basicConfig(level=logging.ERROR)

    run_in_parallel = int(sys.argv[1])
    if run_in_parallel == None:
        logging.error('How many processes in parallel? -> usage: ocr_tesseract [number of parallel processes]')
        sys.exit()


    datapath_in = os.path.join("../data","000_Raw_Images","cropped_tables")
    datapath_out = os.path.join("../data", "002_OCR_all_tables")
    if not os.path.exists(datapath_out):
        os.makedirs(datapath_out)
    years = os.listdir(datapath_in)
    experiments = []
    for year in years:
        if not os.path.exists(os.path.join(datapath_out,year)):
            os.mkdir(os.path.join(datapath_out,year))

        file_list = [tmp for tmp in os.listdir(os.path.join(datapath_in, year)) if not tmp.endswith('.txt')]
        for file_name in file_list:
            if not os.path.exists(os.path.join(datapath_out, year,file_name)):
                os.mkdir(os.path.join(datapath_out, year,file_name))

            tables = [tmp for tmp in os.listdir(os.path.join(datapath_in, year, file_name)) if tmp.endswith('.jpg')]
            for table in tables:
                experiments.append((datapath_in,datapath_out,year,file_name,table))


    p = Pool(run_in_parallel,initializer=mute)
    p.starmap(ocr_tesseract.do_OCR_blockwise_multiprocessing, experiments)

    logging.info('All ocr processes done ::: ')