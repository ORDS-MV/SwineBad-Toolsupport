#Evaluate structuring from GT

import sys,os,json,re
import numpy as np
sys.path.insert(1, '../util')

from evaluation import structure_evaluate


def evaluate_structuring_GT(datapath_annotation,datapath,years,evaluate_data,plot_result = False,enable_output = False):
    """
    Function which evaluates the structuring performance on the groundtruth
    """
    fscore_years = [[] for _ in range(len(years))]
    for year_idx,year in enumerate(years):
        file_list = [tmp for tmp in os.listdir(os.path.join(datapath, year)) if (not tmp.endswith('.txt') and not tmp.startswith('.')) ]
        for file_name in file_list:
            tables = list(set(["".join(tmp.split("_", 3)[-1]) for tmp in os.listdir(os.path.join(datapath, year, file_name)) if tmp.startswith('structuring_on_GT') and not tmp.startswith('.')]))
            fscore_tables = []
            for table in tables:
    
                # load GT
                annotation_file = os.path.join(datapath_annotation, year, file_name,table.strip('.jpg')+'_annotation.json')
                with open(annotation_file, 'r') as f:
                    ground_truth_data = json.load(f)

                num_not_included = 0
                for element in ground_truth_data['imageData']:
                    if element.get('include') == 'False':
                        num_not_included +=1
                    else:
                        break
                
                
                # load indices used for ocr
                with open(os.path.join(datapath_annotation, year, file_name,table.replace('.jpg','')+'_datasplit',evaluate_data,evaluate_data+'_indices_data_dict.txt'), 'r') as f:
                    input_indices = f.read()
                input_indices = re.sub(' +', ' ', input_indices)
                input_indices = np.array(re.sub(' +', ' ', input_indices.replace('[', '').replace(']', '').replace(',', '').replace('\n','')).lstrip().split(' ')).astype(int)

                
                # load structuring result
                with open(os.path.join(datapath,year,file_name,'structuring_on_GT_'+table.strip('.jpg'),'structured_elements'+'.json'),'r') as f:
                    prediction = json.load(f)
    
    
                ground_truth_list = []
                prediction_list = []
                for pred_idx,gt_idx in enumerate(input_indices):
                    gt_tmp = ground_truth_data['imageData'][gt_idx]
                    if gt_tmp.get('include') == 'False':
                        continue
                        
                    gt_tmp.pop('input')
                    try:
                        gt_tmp.pop('Nummer')   
                    except:
                        pass
                    ground_truth_list.append(gt_tmp)
    
                    pred_tmp = prediction[gt_idx-num_not_included]
                    try:
                        pred_tmp.pop('Nummer')   
                    except:
                        pass
    
                    #check for integers
                    count = pred_tmp.get('Personenanzahl')
                    if isinstance(count, int)  :
                        pred_tmp['Personenanzahl'] = str(count)
                    
                    prediction_list.append(pred_tmp)
                    
                    
                f1, _ = structure_evaluate(ground_truth_list, prediction_list,enable_output)
                fscore_tables.append(f1)
        fscore_years[year_idx] = np.mean(fscore_tables)       

    if plot_result:
        import matplotlib.pyplot as plt 
        
        fig = plt.figure(figsize= (20,5))
        gs = fig.add_gridspec(1, 2, wspace=0,width_ratios=[4, 1])
        (ax1, ax2) = gs.subplots(sharex='col', sharey='row')
        x_axis = np.arange(0,len(years),dtype=int)
        
        
        ax1.plot(x_axis,fscore_years)
        ax1.set(xlabel = 'years',ylabel='f1 score')
        ax1.set_xticks(x_axis, years,rotation='vertical')
        
        ax2.boxplot(fscore_years, showmeans=True)
        ax2.set_xticks([1], ['f1'],rotation='vertical')
        fig.savefig('structuring_GT.png', dpi=fig.dpi)
    
    return fscore_years

def evaluate_structuring_OCR_corrected(datapath_annotation,datapath,years,evaluate_data,plot_result = False,enable_output=False):
    """
    Function which evaluates the structuring performance on OCR correction (End-To-End)
    """

    fscore_years_OCR = [[] for _ in range(len(years))]
    for year_idx,year in enumerate(years):
        file_list = [tmp for tmp in os.listdir(os.path.join(datapath, year)) if (not tmp.endswith('.txt') and not tmp.startswith('.')) ]
        for file_name in file_list:
           
            tables = list(set(["".join(tmp.split("_", 4)[-1]) for tmp in os.listdir(os.path.join(datapath, year, file_name)) if tmp.startswith('structuring_OCR_corrected_data') and not tmp.startswith('.')]))
            #tables = ['table_2']
            fscore_tables = []
            for table in tables:
    
                # load GT
                annotation_file = os.path.join(datapath_annotation, year, file_name,table.strip('.jpg')+'_annotation.json')
                with open(annotation_file, 'r') as f:
                    ground_truth_data = json.load(f)
    
                num_not_included = 0
                for element in ground_truth_data['imageData']:
                    if element.get('include') == 'False':
                        num_not_included +=1
                    else:
                        break
                
                
                # load indices used for ocr
                with open(os.path.join(datapath_annotation, year, file_name,table.replace('.jpg','')+'_datasplit',evaluate_data,evaluate_data+'_indices_data_dict.txt'), 'r') as f:
                    input_indices = f.read()
                input_indices = re.sub(' +', ' ', input_indices)
                input_indices = np.array(re.sub(' +', ' ', input_indices.replace('[', '').replace(']', '').replace(',', '').replace('\n','')).lstrip().split(' ')).astype(int)
    
                
                # load structuring result
                with open(os.path.join(datapath,year,file_name,'structuring_OCR_corrected_data_'+table.strip('.jpg'),'structured_elements'+'.json'),'r') as f:
                    prediction = json.load(f)
    
    
                ground_truth_list = []
                prediction_list = []
                for pred_idx,gt_idx in enumerate(input_indices):
                    gt_tmp = ground_truth_data['imageData'][gt_idx]
                    if gt_tmp.get('include') == 'False':
                        continue
                        
                    gt_tmp.pop('input')
                    try:
                        gt_tmp.pop('Nummer')   
                    except:
                        pass
                    ground_truth_list.append(gt_tmp)
    
    
                    #pred_tmp = prediction[pred_idx]
                    pred_tmp = prediction[gt_idx-num_not_included]
                    try:
                        pred_tmp.pop('Nummer')   
                    except:
                        pass
    
                    #check for integers
                    count = pred_tmp.get('Personenanzahl')
                    if isinstance(count, int)  :
                        pred_tmp['Personenanzahl'] = str(count)
                    
                    prediction_list.append(pred_tmp)
                    
                    
                f1, _ = structure_evaluate(ground_truth_list, prediction_list,enable_output)
                fscore_tables.append(f1)
        fscore_years_OCR[year_idx] = np.mean(fscore_tables)        

    if plot_result:
        import matplotlib.pyplot as plt 
        
        fig = plt.figure(figsize= (20,5))
        gs = fig.add_gridspec(1, 2, wspace=0,width_ratios=[4, 1])
        (ax1, ax2) = gs.subplots(sharex='col', sharey='row')
        x_axis = np.arange(0,len(years),dtype=int)
        
        
        ax1.plot(x_axis,fscore_years_OCR)
        ax1.set(xlabel = 'years',ylabel='f1 score')
        ax1.set_xticks(x_axis, years,rotation='vertical')
        
        ax2.boxplot(fscore_years, showmeans=True)
        ax2.set_xticks([1], ['f1'],rotation='vertical')
        fig.savefig('structuring_end_to_end.png', dpi=fig.dpi)
    
    return fscore_years_OCR

if __name__ == "__main__":

    datapath_annotation = '../../data/001_Annotation'
    datapath_structuring = '../../data/005_Structuring_evaluation'

    years = os.listdir(datapath_annotation)
    evaluate_data = 'train'

    evaluate_GT = False
    if evaluate_GT:
        evaluate_structuring_GT(datapath_annotation,datapath_structuring,years,evaluate_data)
    else:
        evaluate_structuring_OCR_corrected(datapath_annotation,datapath_structuring,years,evaluate_data)




     