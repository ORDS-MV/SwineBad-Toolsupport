import os
import numpy as np
import random
import json
import shutil

random.seed(42)
np.random.seed(42)

def get_split_indices(data_dict,split):
    """
       Determine train,validation and test indices

       Parameters
       ----------
       data_dict : dictionary of the annotation file
       split : (train,valid,test) tuple, values in percent

       Returns
       ----------
       train,validation and test indices
       """
    n_lines = len(data_dict)
    indices_data_dict = np.arange(0,n_lines , dtype=int)
    random.shuffle(indices_data_dict)

    train_end = int(n_lines * (split[0] / 100))
    validation_end = train_end + int(n_lines * (split[1] / 100))

    train_indices = indices_data_dict[:train_end]
    validation_indices = indices_data_dict[train_end:validation_end]
    test_indices = indices_data_dict[validation_end:]

    return train_indices,validation_indices, test_indices

def get_lines_from_file(ocr_file, indices):
    """
       Extract specific lines from OCR file

       Parameters
       ----------
       ocr_file : String OCR file
       indices : list of indices to be extracted

       Returns
       ----------
       Desired lines extracted from OCR file
       """
    selected_lines = []
    lines = ocr_file.splitlines()
    for index in indices:
        if 0 <= index < len(lines):
            selected_lines.append(lines[index].strip())
    return selected_lines


def prepare_dataset(datapath,years,split):
    """
       Prepare the dataset such that it will be split in train, validation and
       test set. This is done for every table of the specific years.

       Parameters
       ----------
       datapath : Path to the directory containing the OCR files of all years
       years : list of years to be processed (years must be string)
       split : (train,valid,test) tuple, values in percent

       Note:
       ----------
       Stores the indices corresponding to the annotation file in train/test/validation_indices_data.
       Stores the indices corresponding to the ocr file in train/test/validation_indices_ocr.
       Stores the connection between annotation data and ocr data in index_connection.json
       """


    for year in years:
        file_list = [tmp for tmp in os.listdir(os.path.join(datapath, year)) if not tmp.endswith('.txt')]
        for file_name in file_list:
            tables = [tmp for tmp in os.listdir(os.path.join(datapath, year, file_name)) if tmp.endswith('.jpg')]
            for table in tables:

                if os.path.exists(os.path.join(datapath, year, file_name,table.strip('.jpg')+'_datasplit')):
                    shutil.rmtree(os.path.join(datapath, year, file_name,table.strip('.jpg')+'_datasplit'))
                os.makedirs(os.path.join(datapath, year, file_name,table.strip('.jpg')+'_datasplit'))
                os.makedirs(os.path.join(datapath, year, file_name, table.strip('.jpg') + '_datasplit','train'))
                os.makedirs(os.path.join(datapath, year, file_name, table.strip('.jpg') + '_datasplit', 'validation'))
                os.makedirs(os.path.join(datapath, year, file_name, table.strip('.jpg') + '_datasplit', 'test'))

                # get indices from annotation file and split to test and train data
                annotation_file = os.path.join('../../data/001_Annotation', year, file_name, table.strip('.jpg') + '_annotation.json')
                with open(annotation_file, 'r') as f:
                    ground_truth_data = json.load(f)

                line_index_connection = {}
                ocr_line_ptr = 0
                for annotation_idx,annotation in enumerate(ground_truth_data.get('imageData')):
                    line = annotation.get('input')
                    tmp = [ocr_line_ptr]
                    line_index_connection[annotation_idx] = tmp
                    ocr_line_ptr += 1
                    if '\n' in line:
                        line_index_connection.get(annotation_idx).append(ocr_line_ptr)
                        ocr_line_ptr+=1


                with open(os.path.join(datapath, year, file_name, table.strip('.jpg') + '_index_connection.json'), 'w') as f:
                    json.dump(line_index_connection, f, ensure_ascii=True)

                train_indices_data_dict,validation_indices_data_dict, test_indices_data_dict = get_split_indices(ground_truth_data.get('imageData'),split)

                train_indices_ocr_data = []
                for idx in train_indices_data_dict:
                    train_indices_ocr_data.append(line_index_connection.get(idx))
                train_indices_ocr_data = [x for xs in train_indices_ocr_data for x in xs]

                validation_indices_ocr_data = []
                for idx in validation_indices_data_dict:
                    validation_indices_ocr_data.append(line_index_connection.get(idx))
                validation_indices_ocr_data = [x for xs in validation_indices_ocr_data for x in xs]

                test_indices_ocr_data = []
                for idx in test_indices_data_dict:
                    test_indices_ocr_data.append(line_index_connection.get(idx))
                test_indices_ocr_data = [x for xs in test_indices_ocr_data for x in xs]


                # store indices
                with open(os.path.join(datapath, year, file_name, table.strip('.jpg') + '_datasplit','train', 'train_indices_data_dict.txt'), 'w') as f:
                    f.write(str(train_indices_data_dict))
                with open(os.path.join(datapath, year, file_name, table.strip('.jpg') + '_datasplit','validation','validation_indices_data_dict.txt'), 'w') as f:
                    f.write(str(validation_indices_data_dict))
                with open(os.path.join(datapath, year, file_name, table.strip('.jpg') + '_datasplit','test','test_indices_data_dict.txt'), 'w') as f:
                    f.write(str(test_indices_data_dict))
                with open(os.path.join(datapath, year, file_name, table.strip('.jpg') + '_datasplit','train', 'train_indices_ocr_file.txt'), 'w') as f:
                    f.write(str(train_indices_ocr_data))
                with open(os.path.join(datapath, year, file_name, table.strip('.jpg') + '_datasplit','validation','validation_indices_ocr_file.txt'), 'w') as f:
                    f.write(str(validation_indices_ocr_data))
                with open(os.path.join(datapath, year, file_name, table.strip('.jpg') + '_datasplit','test','test_indices_ocr_file.txt'), 'w') as f:
                    f.write(str(test_indices_ocr_data))


if __name__ == "__main__":

    datapath = '../../data/001_Annotation'
    years = os.listdir(datapath)
    split = (60,20,20)
    prepare_dataset(datapath, years,split)