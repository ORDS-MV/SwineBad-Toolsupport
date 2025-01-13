import sys
import os
import numpy as np
import re
import json
import Levenshtein

sys.path.insert(1, '../util')
from evaluation import evaluate_OCR
from alignment import align_blocks


def evaluate_corrected_ocr(datapath_annotation,datapath,years,evaluate_data):
    """
    Function which evaluates the corrected OCR performance line by line. Stores a direct comparison to the GT
    """
    
    years = sorted(years)
    for year in years:
    
        file_list = [tmp for tmp in os.listdir(os.path.join(datapath_annotation, year)) if not tmp.endswith('.txt')]
        #file_list = ['00000001']
        for file_name in file_list:
    
            tables = [tmp for tmp in os.listdir(os.path.join(datapath_annotation, year, file_name)) if tmp.endswith('.jpg')]
            #tables = ['table_1.jpg']
            for table in tables:
    
                os.makedirs(os.path.join(datapath, year, file_name,'ocr_correction_'+table.strip('.jpg') ,'evaluation'),exist_ok=True)
    
                full_corrected_ocr_files = [tmp for tmp in os.listdir(os.path.join(datapath, year, file_name,'ocr_correction_'+table.strip('.jpg'),'correction')) if tmp.endswith('.txt')]
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
    
    
                for ocr_file in full_corrected_ocr_files:
                    with open(os.path.join(datapath, year, file_name,'ocr_correction_'+table.strip('.jpg'),'correction',ocr_file), 'r') as f:
                        full_corrected_ocr_data = f.read()
    
                    full_corrected_ocr_data = os.linesep.join([s for s in full_corrected_ocr_data.splitlines() if s])
                    full_corrected_ocr_data = full_corrected_ocr_data.splitlines()
                    full_gt_data = ground_truth_data_string.splitlines()
    
                    # alignment:
                    # GT is always complete
                    # case 1: both ocr and gt are complete. all works fine
                    # case 2: ocr has missed a line -> it is filled with an empty line
                    # case 3: ocr has produced a line which cannot be aligned with the gt
                    #       -> the gt is filled up with empty lines at the end
                    #       -> can be skipped, since no truth data cannot calculate Levenshtein distance
                    #       -> Since the error causes an empty row in the ocr data, it produces max error (CER=1) in evaluation
    
                    aligned_gt_data,aligned_ocr_data = align_blocks(full_gt_data, full_corrected_ocr_data, None,15)  # TODO threshold korrekt?
                    
                    gt_split_data = np.array(aligned_gt_data)[input_indices]
                    ocr_split_data = np.array(aligned_ocr_data)[input_indices]
    
    
                    eval_string = evaluate_OCR(ocr_split_data,gt_split_data)
    
    
                    with open(os.path.join(datapath, year, file_name,'ocr_correction_'+table.strip('.jpg'),'evaluation',ocr_file.replace('.txt','')+'_'+evaluate_data+'_evaluation.txt'), 'w') as f:
                        for line in eval_string:
                            f.write(f"{line}\n")

def summarize_ocr_correction_result(datapath_annotation,datapath,years,evaluate_data,plot_result = False):
    """
    Function summarizes the evaluation files for corrected OCR created by evaluate_corrected_ocr() in order to do plots
    """
    result_years = [[] for _ in range(len(years))]
    for idx,year in enumerate(years):
        result_dict = {'123':[],'132':[],'213':[],'231':[],'312':[],'321':[]}
        file_list = [tmp for tmp in os.listdir(os.path.join(datapath_annotation, year)) if not tmp.endswith('.txt')]
        for file_name in file_list:
            tables = [tmp for tmp in os.listdir(os.path.join(datapath_annotation, year, file_name)) if tmp.endswith('.jpg')]
            for table in tables:
                corrected_output_evaluation_files = [tmp for tmp in os.listdir(os.path.join(datapath,year,file_name,'ocr_correction_'+table.strip('.jpg'),'evaluation')) if tmp.endswith('.txt') and evaluate_data in tmp]
                
                for ocr_file in corrected_output_evaluation_files:
                    with open(os.path.join(datapath,year,file_name,'ocr_correction_'+table.strip('.jpg'),'evaluation',ocr_file), 'r') as f:
                        result = f.read()
    
                    for line in result.splitlines():
                        if 'final CER' in line:
                            final_CER = float(line.split('final CER:')[1].split('mean CER:')[0])
    
                    if '123' in ocr_file:
                        result_dict['123'].append(final_CER)
                    if '132' in ocr_file:
                        result_dict['132'].append(final_CER)
                    if '213' in ocr_file:
                        result_dict['213'].append(final_CER)
                    if '231' in ocr_file:
                        result_dict['231'].append(final_CER)
                    if '312' in ocr_file:
                        result_dict['312'].append(final_CER)
                    if '321' in ocr_file:
                        result_dict['321'].append(final_CER)
        result_dict['123'] = np.mean(result_dict.get('123'))
        result_dict['132'] = np.mean(result_dict.get('132'))
        result_dict['213'] = np.mean(result_dict.get('213'))
        result_dict['231'] = np.mean(result_dict.get('231'))
        result_dict['312'] = np.mean(result_dict.get('312'))
        result_dict['321'] = np.mean(result_dict.get('321'))
        result_years[idx] = result_dict
    
    
    order_123_res = []
    order_132_res = []
    order_213_res = []
    order_231_res = []
    order_312_res = []
    order_321_res = []
    for idx,year in enumerate(years):
        file_list = [tmp for tmp in os.listdir(os.path.join(datapath_annotation, year)) if not tmp.endswith('.txt')]
        
        order_123_res.append(result_years[idx].get('123'))
        order_132_res.append(result_years[idx].get('132'))
        order_213_res.append(result_years[idx].get('213'))
        order_231_res.append(result_years[idx].get('231'))
        order_312_res.append(result_years[idx].get('312'))
        order_321_res.append(result_years[idx].get('321'))
    
        with open(os.path.join(datapath, year, file_list[0],'ocr_results_corrected.txt'), 'w') as f:
            f.write('mean_CER_123: '+str(result_years[idx].get('123'))+'\n'
                            +'mean_CER_132: '+str(result_years[idx].get('132'))+'\n'
                            +'mean_CER_213: '+str(result_years[idx].get('213'))+'\n'
                            +'mean_CER_231: '+str(result_years[idx].get('231'))+'\n'
                            +'mean_CER_312: '+str(result_years[idx].get('312'))+'\n'
                            +'mean_CER_321: '+str(result_years[idx].get('321')))
    if plot_result:
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(10,3))
        ax = fig.add_subplot(111)
        #ax = plt.subplot(111)
        x_axis = np.arange(len(years))
        ax.bar(x_axis-0.2,order_123_res,0.1,label='123')
        ax.bar(x_axis-0.1,order_132_res,0.1,label='132')
        ax.bar(x_axis,order_213_res,0.1,label='213')
        ax.bar(x_axis+0.1,order_231_res,0.1,label='231')
        ax.bar(x_axis+0.2,order_312_res,0.1,label='312')
        ax.bar(x_axis+0.3,order_321_res,0.1,label='321')
        plt.xlabel('years')
        plt.ylabel('mean_CER')
        plt.xticks(x_axis, years,rotation='vertical')
        plt.legend()
        plt.show()
        fig.savefig('OCR_correction_evaluation_orders.png', dpi=fig.dpi)

    return order_123_res,order_132_res,order_213_res,order_231_res,order_312_res,order_321_res

if __name__ == "__main__":

    datapath_annotation = '../../data/001_Annotation'
    datapath_corrected_OCR_files = '../../data/004_OCR_correction_evaluation'
    years = os.listdir(datapath_annotation)
    
    evaluate_data = 'train'

    evaluate_corrected_ocr(datapath_annotation,datapath_corrected_OCR_files,years,evaluate_data)
    summarize_ocr_correction_result(datapath_annotation,datapath_corrected_OCR_files,years,evaluate_data,True)

