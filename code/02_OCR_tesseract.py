import sys

import cv2
import pytesseract
import os
import numpy as np
import re


if sys.platform == 'win32':
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


def blockwise_OCR(model,table_path):
    """
    Perform blockwise OCR with tesseract for a specific image
    """

    img = cv2.imread(table_path)

    # prepare image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, img_bin) = cv2.threshold(img_gray, 210, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    tesseract_config = '--psm 6 --oem 3 --tessdata-dir "./../OCR_Models"'
    pattern = '[^ „a-zA-Z0-9.,⸗"|äöüß-]'

    tess_data = pytesseract.image_to_data(img_bin, config=tesseract_config, lang=model,
                                                   output_type=pytesseract.Output.DICT)
    ocr_string = pytesseract.image_to_string(img_bin, config=tesseract_config, lang=model,
                              output_type=pytesseract.Output.DICT)

    lines = [[] for _ in range(np.max(tess_data['line_num']) + 1)]
    conf = [[] for _ in range(np.max(tess_data['line_num']) + 1)]
    for txt_idx, txt in enumerate(tess_data['text']):
        lines[tess_data['line_num'][txt_idx]].append(txt)
        conf[tess_data['line_num'][txt_idx]].append(tess_data['conf'][txt_idx])
    conf_per_line = [np.mean(conf[x]) for x in range(len(conf))]

    num_lines = 0
    result = ""
    for line_idx, line_str in enumerate(ocr_string['text'].splitlines()):
        # \u017f is LATIN SMALL LETTER LONG S, used in Fraktur -> replace with s
        line_str = line_str.replace('\u017f', 's')
        line_str = re.sub(pattern, '', line_str)
        result += line_str + '\n'
        num_lines += 1

    return result,np.mean(conf_per_line)


def do_OCR_blockwise(datapath, years):
    """
    Perform blockwise OCR for all images in the datapath and stores the results there
    """
    if not isinstance(years, list):
        years = [years]
    for year in years:
        file_list = [tmp for tmp in os.listdir(os.path.join(datapath, year)) if not tmp.endswith('.txt')]
        for file_name in file_list:
            tables = [x for x in os.listdir(os.path.join(datapath, year, file_name)) if x.endswith('.jpg')]
            for table in tables:
                if table.endswith('.jpg'):
                    table_path = os.path.join(datapath, year, file_name, table)

                    models = ['frak2021', 'GT4HistOCR', 'Fraktur']

                    for model in models:

                        # skip if already existing
                        if table.replace('.jpg', '') + "_ocr_" + model + ".txt" in os.listdir(
                                os.path.join(datapath, year, file_name)):
                            continue

                        result_blockwise, mean_conf_blockwise = blockwise_OCR(model,table_path)

                        with open(os.path.join(datapath, year, file_name,table.replace('.jpg', '') + "_ocr_" + model + ".txt"), "w") as file:
                            file.write(result_blockwise)

def do_OCR_blockwise_multiprocessing(datapath_in,datapath_out, year,file_name,table):
    """
        Perform blockwise OCR for all images in the datapath_in
        Store results in datapath_out

        Note: This function is applied in do_OCR_multiprocessing.py
    """

    table_path = os.path.join(datapath_in, year, file_name, table)
    models = ['frak2021', 'GT4HistOCR', 'Fraktur']

    for model in models:

        result_blockwise, mean_conf_blockwise = blockwise_OCR(model,table_path)

        with open(os.path.join(datapath_out, year, file_name,table.replace('.jpg', '') + "_ocr_" + model + ".txt"), "w") as file:
            file.write(result_blockwise)


if __name__ == "__main__":

    datapath = '../data/000_Raw_Images/cropped_tables'
    years = os.listdir(datapath)
    do_OCR_blockwise(datapath, years)







