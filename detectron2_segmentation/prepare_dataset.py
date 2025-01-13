"""
Script to prepare the dataset to fine-tune detectron2
Splits the data in train, test and validate data
"""
import json
import os
import shutil
from labelme2coco import get_coco_from_labelme_folder, save_json
from sklearn.model_selection import train_test_split
import config


years = os.listdir(config.GROUNDTRUTH_BASE_DATASET+'/images')
files_all_years = []
for year in years:
    # list all files in a single list
    for file in os.listdir(os.path.join(config.GROUNDTRUTH_BASE_DATASET,'images',year)):
        files_all_years.append(file.split('.')[0])


# create directories for train, test and validation data - remove already existing
if os.path.exists(config.BASE_DATASET):
    shutil.rmtree(config.BASE_DATASET)
os.makedirs(config.BASE_DATASET)
os.makedirs(config.TRAIN_DATASET_PATH)
os.makedirs(config.TEST_DATASET_PATH)
os.makedirs(config.VALID_DATASET_PATH)


# partition the data into training, validate and testing splits
assert config.DATA_SPLIT.get('train')+config.DATA_SPLIT.get('validate')+config.DATA_SPLIT.get('test') == 1 , "Splits must sum up to 1"

if config.DATA_SPLIT.get('validate') == 0.0:
    train_images, test_images = train_test_split(files_all_years, train_size=config.DATA_SPLIT.get('train'),random_state=42)
    validate_images = []
else:
    train_images,further_images = train_test_split(files_all_years,train_size=config.DATA_SPLIT.get('train'), random_state=42)
    validate_split_value = config.DATA_SPLIT.get('validate') / (1-config.DATA_SPLIT.get('train'))
    validate_images,test_images = train_test_split(further_images,train_size=validate_split_value, random_state=42)


# write the lists to disk
with open(config.TRAIN_DATASET_PATH+'/train_data.txt', "w") as f:
    f.write("\n".join(train_images))
with open(config.VALID_DATASET_PATH+'/valid_data.txt', "w") as f:
    f.write("\n".join(validate_images))
with open(config.TEST_DATASET_PATH+'/test_data.txt', "w") as f:
    f.write("\n".join(test_images))


# fill train, test and valid directory with data
for name in train_images:
    year = name.split('_')[0]
    path_to_image = os.path.join(config.GROUNDTRUTH_BASE_DATASET,'images',year,name)
    shutil.copy(path_to_image +'.jpg', 'dataset/train')

    path_to_JSON = os.path.join(config.GROUNDTRUTH_BASE_DATASET,'JSON',year,name)
    shutil.copy(path_to_JSON+'.json', config.TRAIN_DATASET_PATH)

    # update image path in json file
    with open(os.path.join(config.TRAIN_DATASET_PATH,name)+'.json', "r") as jsonFile:
        data = json.load(jsonFile)

    data["imagePath"] = name + '.jpg'

    with open(os.path.join(config.TRAIN_DATASET_PATH,name)+'.json', "w") as jsonFile:
        json.dump(data, jsonFile, ensure_ascii=True, indent=4, sort_keys=True)


for name in test_images:
    year = name.split('_')[0]
    path_to_image = os.path.join(config.GROUNDTRUTH_BASE_DATASET,'images',year,name)
    shutil.copy(path_to_image+'.jpg', 'dataset/test')

    path_to_JSON = os.path.join(config.GROUNDTRUTH_BASE_DATASET,'JSON',year,name)
    shutil.copy(path_to_JSON+'.json', config.TEST_DATASET_PATH)

    # update image path in json file
    with open(os.path.join(config.TEST_DATASET_PATH, name) + '.json', "r") as jsonFile:
        data = json.load(jsonFile)

    data["imagePath"] = name + '.jpg'

    with open(os.path.join(config.TEST_DATASET_PATH, name) + '.json', "w") as jsonFile:
        json.dump(data, jsonFile, ensure_ascii=True, indent=4, sort_keys=True)


for name in validate_images:
    year = name.split('_')[0]
    path_to_image = os.path.join(config.GROUNDTRUTH_BASE_DATASET,'images',year,name)
    shutil.copy(path_to_image+'.jpg', 'dataset/valid')

    path_to_JSON = os.path.join(config.GROUNDTRUTH_BASE_DATASET,'JSON',year,name)
    shutil.copy(path_to_JSON+'.json', config.VALID_DATASET_PATH)

    # update image path in json file
    with open(os.path.join(config.VALID_DATASET_PATH,name)+'.json', "r") as jsonFile:
        data = json.load(jsonFile)

    data["imagePath"] = name + '.jpg'

    with open(os.path.join(config.VALID_DATASET_PATH,name)+'.json', "w") as jsonFile:
        json.dump(data, jsonFile, ensure_ascii=True, indent=4, sort_keys=True)


# set category ID start value
category_id_start = 1

# create coco objects
train_coco = get_coco_from_labelme_folder(config.TRAIN_DATASET_PATH, category_id_start=category_id_start)
test_coco = get_coco_from_labelme_folder(config.TEST_DATASET_PATH, category_id_start=category_id_start)
valid_coco = get_coco_from_labelme_folder(config.VALID_DATASET_PATH, category_id_start=category_id_start)

# export coco json
save_json(train_coco.json, config.TRAIN_DATASET_PATH+"/train.json")
save_json(test_coco.json, config.TEST_DATASET_PATH+"/test.json")
save_json(valid_coco.json, config.VALID_DATASET_PATH+"/valid.json")

