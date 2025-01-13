import sys
import os
import numpy as np
import re
import json
import Levenshtein

sys.path.insert(1, '../util')
from evaluation import evaluate_OCR
from alignment import align_blocks


def evaluate_ocr(datapath_annotation,datapath_ocr_files,years,evaluate_data):
    """
    Function which evaluates the OCR performance line by line. Stores a direct comparison to the GT
    """


    if not os.path.exists('../../data/003_OCR_evaluation'):
        os.makedirs('../../data/003_OCR_evaluation')

    years = sorted(years)
    for year in years:

        if not os.path.exists(os.path.join('../../data/003_OCR_evaluation', year)):
            os.makedirs(os.path.join('../../data/003_OCR_evaluation', year))

        file_list = [tmp for tmp in os.listdir(os.path.join(datapath_annotation, year)) if not tmp.endswith('.txt')]
        for file_name in file_list:

            if not os.path.exists(os.path.join('../../data/003_OCR_evaluation', year, file_name)):
                os.makedirs(os.path.join('../../data/003_OCR_evaluation', year, file_name))

            tables = [tmp for tmp in os.listdir(os.path.join(datapath_annotation, year, file_name)) if tmp.endswith('.jpg')]
            for table in tables:

                os.makedirs(os.path.join('../../data/003_OCR_evaluation', year, file_name, table.strip('.jpg') + '_evaluation'), exist_ok=True)

                full_ocr_files = [tmp for tmp in os.listdir(os.path.join(datapath_ocr_files, year, file_name)) if tmp.endswith('.txt') and table.strip('.jpg') in tmp]
                annotation_file = os.path.join(datapath_annotation, year, file_name, table.strip('.jpg') + '_annotation.json')

                # prepare gt data
                with open(annotation_file, 'r') as f:
                    ground_truth_data = json.load(f)
                ground_truth_data_string = ""
                for data_file in ground_truth_data.get("imageData"):
                    line = data_file.get('input')
                    for split in line.split('\n'):
                        ground_truth_data_string+= split+'\n'


                # load indices used for ocr
                with open(os.path.join(datapath_annotation, year, file_name,table.replace('.jpg','')+'_datasplit',evaluate_data,evaluate_data+'_indices_ocr_file.txt'), 'r') as f:
                    input_indices = f.read()
                input_indices = np.array(input_indices.replace('[', '').replace(']', '').replace(',', '').split(' ')).astype(int)


                for ocr_file in full_ocr_files:
                    with open(os.path.join(datapath_ocr_files, year, file_name,ocr_file), 'r') as f:
                        full_ocr_data = f.read()

                    full_ocr_data = os.linesep.join([s for s in full_ocr_data.splitlines() if s])
                    full_ocr_data = full_ocr_data.splitlines()
                    full_gt_data = ground_truth_data_string.splitlines()

                    # alignment:
                    # GT is always complete
                    # case 1: both ocr and gt are complete. all works fine
                    # case 2: ocr has missed a line -> it is filled with an empty line
                    # case 3: ocr has produced a line which cannot be aligned with the gt
                    #       -> the gt is filled up with empty lines at the end
                    #       -> can be skipped, since no truth data cannot calculate Levenshtein distance
                    #       -> Since the error causes an empty row in the ocr data, it produces max error (CER=1) in evaluation

                    aligned_gt_data,aligned_ocr_data = align_blocks(full_gt_data, full_ocr_data, 10,30)  # TODO threshold korrekt?
                    gt_split_data = np.array(aligned_gt_data)[input_indices]
                    ocr_split_data = np.array(aligned_ocr_data)[input_indices]


                    eval_string = evaluate_OCR(ocr_split_data,gt_split_data)


                    with open(os.path.join('../../data/003_OCR_evaluation', year, file_name, table.strip('.jpg') + '_evaluation', ocr_file.replace('.txt', '') + '_' + evaluate_data + '_evaluation.txt'), 'w') as f:
                        for line in eval_string:
                            f.write(f"{line}\n")


def summarize_ocr_result(datapath_ocr_evaluation,years,evaluate_data,plot_result = False):
    """
    Function summarizes the evaluation files created by evaluate_ocr() in order to do plots
    """
    years = sorted(years)
    result_years = [[] for _ in range(len(years))]
    for idx,year in enumerate(years):
        result_dict = {'frak2021':[],'GT4HistOCR':[],'Fraktur':[]}
        file_list = [tmp for tmp in os.listdir(os.path.join(datapath_ocr_evaluation, year)) if not tmp.startswith('.') and not tmp.endswith('.txt')]

        for file_name in file_list:
            tables = list(set(["_".join(tmp.split("_", 2)[:2]) for tmp in os.listdir(os.path.join(datapath_ocr_evaluation, year, file_name))]))
            for table in tables:
                ocr_evaluation_files = [tmp for tmp in os.listdir(
                    os.path.join(datapath_ocr_evaluation, year, file_name, table + '_evaluation')) if
                             tmp.endswith('.txt') and table.strip('.jpg') in tmp and evaluate_data in tmp]

                for ocr_file in ocr_evaluation_files:
                    with open(os.path.join(datapath_ocr_evaluation, year, file_name, table+ '_evaluation',
                                           ocr_file), 'r') as f:
                        result = f.read()

                    for line in result.splitlines():
                        if 'final CER' in line:
                            final_CER = float(line.split('final CER:')[1].split('mean CER:')[0])

                    if 'frak2021' in ocr_file:
                        result_dict['frak2021'].append(final_CER)
                    if 'GT4HistOCR' in ocr_file:
                        result_dict['GT4HistOCR'].append(final_CER)
                    if 'Fraktur' in ocr_file:
                        result_dict['Fraktur'].append(final_CER)
        result_dict['frak2021'] = np.mean(result_dict.get('frak2021'))
        result_dict['GT4HistOCR'] = np.mean(result_dict.get('GT4HistOCR'))
        result_dict['Fraktur'] = np.mean(result_dict.get('Fraktur'))
        result_years[idx] = result_dict

    
    frak2021_res = []
    Fraktur_res = []
    GT4HistOCR_res = []

    for idx,year in enumerate(years):
        frak2021_res.append(result_years[idx].get('frak2021'))
        Fraktur_res.append(result_years[idx].get('Fraktur'))
        GT4HistOCR_res.append(result_years[idx].get('GT4HistOCR'))

        with open(os.path.join(datapath_ocr_evaluation, year, 'ocr_results.txt'), 'w') as f:
            f.write('mean_CER_frak2021: '+str(result_years[idx].get('frak2021'))+'\n'
                            +'mean_CER_GT4HistOCR: '+str(result_years[idx].get('GT4HistOCR'))+'\n'
                            +'mean_CER_Fraktur: '+str(result_years[idx].get('Fraktur')))
    if plot_result:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        x_axis = np.arange(len(years))
        ax.bar(x_axis+0.2,frak2021_res,0.2,label='frak2021')
        ax.bar(x_axis,Fraktur_res,0.2,label='Fraktur')
        ax.bar(x_axis-0.2,GT4HistOCR_res,0.2,label='GT4HistOCR')
        plt.xticks(x_axis, years,rotation='vertical')
        plt.xlabel('years')
        plt.ylabel('mean_CER')
        plt.tight_layout()
        plt.legend()
        plt.show()
        fig.savefig('OCR_evaluation.png', dpi=fig.dpi)
    return frak2021_res,Fraktur_res,GT4HistOCR_res


if __name__ == "__main__":

    datapath_annotation = '../../data/001_Annotation'
    datapath_OCR_files = '../../data/002_OCR_all_tables'
    datapath_OCR_evaluation = '../../data/003_OCR_evaluation'

    years = os.listdir(datapath_annotation)
    evaluate_data = 'train'

    evaluate_ocr(datapath_annotation,datapath_OCR_files,years,evaluate_data)
    summarize_ocr_result(datapath_OCR_evaluation,years,evaluate_data,True)
