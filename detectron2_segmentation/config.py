import torch
import os


# define paths
# GROUNDTRUTH_BASE_DATASET = os.path.join('..','groundtruth')
# GROUNDTRUTH_BASE_DATASET = os.path.join('..','groundtruth_low_res')
GROUNDTRUTH_BASE_DATASET = os.path.join('..','groundtruth_low_res_1910')
BASE_DATASET = os.path.join(os.getcwd(),'dataset')
TRAIN_DATASET_PATH = os.path.join(BASE_DATASET, "train")
TEST_DATASET_PATH = os.path.join(BASE_DATASET, "test")
VALID_DATASET_PATH = os.path.join(BASE_DATASET, "valid")

BASE_OUTPUT = os.path.join(os.getcwd(),'output')

BASE_PRETRAINED_MODEL = os.path.join(os.getcwd(),'pretrained_model')
PRETRAINED_MODEL_YAML_PATH = os.path.join(BASE_PRETRAINED_MODEL, "TableBank_X152.yaml")
PRETRAINED_MODEL_PTH_PATH = os.path.join(BASE_PRETRAINED_MODEL, "TableBank_X152.pth")

# datasets
NAME_TRAIN_DATASET = "asl_poly_train"
NAME_TEST_DATASET = "asl_poly_test"
NAME_VALID_DATASET = "asl_poly_valid"

# define the test split
DATA_SPLIT = {'train':0.8,'validate':0.0,'test':0.2}

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

#---------other parameters-----------------

# define the input image dimensions
# low resolution
INPUT_MIN_SIZE_TRAIN = (800,)
INPUT_MAX_SIZE_TRAIN = 1333
INPUT_MIN_SIZE_TEST = 800
INPUT_MAX_SIZE_TEST = 1333

# high resolution
# INPUT_MIN_SIZE_TRAIN = (3378,)
# INPUT_MAX_SIZE_TRAIN = 5687
# INPUT_MIN_SIZE_TEST = 3378
# INPUT_MAX_SIZE_TEST = 5687


# define threshold to filter weak predictions
THRESHOLD = 0.5

DATALOADER_NUM_WORKER = 2

BATCH_SIZE = 2
SOLVER_BASE_LR = 0.00025
SOLVER_MAX_ITER = 100000

MODEL_ROI_HEADS_BATCH_SIZE_PER_IMAGE = 512
MODEL_ROI_HEADS_NUM_CLASSES = 1

TEST_EVAL_PERIOD = 0 # Increase this number if you want to monitor validation performance during training

PATIENCE = 100  # Early stopping will occur after N iterations of no imporovement in total_loss