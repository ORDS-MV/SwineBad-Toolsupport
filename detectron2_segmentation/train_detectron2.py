import sys

import logging
import torch
from collections import OrderedDict
import numpy as np
import os, json, cv2, random
from urllib.request import urlopen
import shutil
import tarfile


from detectron2.data.datasets import register_coco_instances
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.utils.events import EventStorage
from detectron2.modeling import build_model
import detectron2.utils.comm as comm
from detectron2.engine import default_writers
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)

# from matplotlib import pyplot as plt
from PIL import Image

import rcnn_model
import config

import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


"""Usage:

If the dataset is not located in the project directory, please use:
train_detectron2 [--data PATH_TO_DATASET]
"""


def download_pretrained_model():
    #download pretrained model if not existing
    if not os.path.exists(os.path.join('pretrained_model','TableBank_X152.pth')):
        pretrained_model_url = '...'
        filepath_pretrained_model = os.path.join('pretrained_model','TableBank_X152.pth')

        try:
            with urlopen(pretrained_model_url) as conn:
                print('Downloading pretrained model...')
                with open(filepath_pretrained_model, 'b+w') as f:
                    f.write(conn.read())
        except Exception:
            print("ERROR downloading pretrained model")

def register_data_sets(individual_dataset_path = None):

    if individual_dataset_path is not None:
        with open(os.path.join(individual_dataset_path,'train','train.json'), "r") as jsonFile:
            data = json.load(jsonFile)
            for idx,image in enumerate(data['images']):
                name = image['file_name']
                image_name = name.split('/')[-1]
                data['images'][idx]['file_name'] = image_name

        with open(os.path.join(individual_dataset_path,'train','train.json'), "w") as jsonFile:
            json.dump(data, jsonFile, ensure_ascii=True, indent=4, sort_keys=True)

        with open(os.path.join(individual_dataset_path,'test','test.json'), "r") as jsonFile:
            data = json.load(jsonFile)
            for idx,image in enumerate(data['images']):
                name = image['file_name']
                image_name = name.split('/')[-1]
                data['images'][idx]['file_name'] = image_name

        with open(os.path.join(individual_dataset_path,'test','test.json'), "w") as jsonFile:
            json.dump(data, jsonFile, ensure_ascii=True, indent=4, sort_keys=True)

        with open(os.path.join(individual_dataset_path,'valid','valid.json'), "r") as jsonFile:
            data = json.load(jsonFile)
            for idx,image in enumerate(data['images']):
                name = image['file_name']
                image_name = name.split('/')[-1]
                data['images'][idx]['file_name'] = image_name

        with open(os.path.join(individual_dataset_path,'valid','valid.json'), "w") as jsonFile:
            json.dump(data, jsonFile, ensure_ascii=True, indent=4, sort_keys=True)

    # register datasets
    if individual_dataset_path == None:
        register_coco_instances(config.NAME_TRAIN_DATASET, {}, config.TRAIN_DATASET_PATH + '/train.json',
                                config.TRAIN_DATASET_PATH)
        register_coco_instances(config.NAME_TEST_DATASET, {}, config.TEST_DATASET_PATH + '/test.json',
                                config.TEST_DATASET_PATH)
        register_coco_instances(config.NAME_VALID_DATASET, {}, config.VALID_DATASET_PATH+'/valid.json',
                                config.VALID_DATASET_PATH)
    else:
        register_coco_instances(config.NAME_TRAIN_DATASET, {}, os.path.join(individual_dataset_path,'train','train.json'),
                                os.path.join(individual_dataset_path,'train'))
        register_coco_instances(config.NAME_TEST_DATASET, {}, os.path.join(individual_dataset_path,'test','test.json'),
                                os.path.join(individual_dataset_path,'test'))
        register_coco_instances(config.NAME_VALID_DATASET, {}, os.path.join(individual_dataset_path,'valid','valid.json'),
                                os.path.join(individual_dataset_path,'valid'))



def prepare_cfg():

    cfg = rcnn_model.setup(config.PRETRAINED_MODEL_YAML_PATH)
    cfg.INPUT.MIN_SIZE_TRAIN = config.INPUT_MIN_SIZE_TRAIN
    cfg.INPUT.MAX_SIZE_TRAIN = config.INPUT_MAX_SIZE_TRAIN
    cfg.INPUT.MIN_SIZE_TEST = config.INPUT_MIN_SIZE_TEST
    cfg.INPUT.MAX_SIZE_TEST = config.INPUT_MAX_SIZE_TEST

    cfg.DATASETS.TRAIN = (config.NAME_TRAIN_DATASET,)
    cfg.DATASETS.TEST = (config.NAME_TEST_DATASET,)
    cfg.DATASETS.VALID = (config.NAME_VALID_DATASET,)
    cfg.DATALOADER.NUM_WORKERS = config.DATALOADER_NUM_WORKER

    cfg.SOLVER.IMS_PER_BATCH = config.BATCH_SIZE
    cfg.SOLVER.BASE_LR = config.SOLVER_BASE_LR
    cfg.SOLVER.MAX_ITER = config.SOLVER_MAX_ITER
    cfg.SOLVER.STEPS = []

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config.MODEL_ROI_HEADS_BATCH_SIZE_PER_IMAGE
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.MODEL_ROI_HEADS_NUM_CLASSES
    cfg.MODEL.DEVICE = config.DEVICE
    cfg.TEST.EVAL_PERIOD = config.TEST_EVAL_PERIOD
    cfg.PATIENCE = config.PATIENCE

    cfg.freeze()

    return cfg

def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def run_training(cfg,training):

    # create output folder
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    BEST_LOSS = np.inf

    resume = False
    model = build_model(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
            checkpointer.resume_or_load('pretrained_model/'+cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )

    if training:
        start_iter = 0 # override
        prev_iter = start_iter
        max_iter = cfg.SOLVER.MAX_ITER

        periodic_checkpointer = PeriodicCheckpointer(
            checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
        )

        writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

        data_loader = build_detection_train_loader(cfg)
        logger.info("Starting training from iteration {}".format(start_iter))
        patience_counter = 0
        with EventStorage(start_iter) as storage:
            for data, iteration in zip(data_loader, range(start_iter, max_iter)):

                storage.iter = iteration

                loss_dict = model(data)
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                scheduler.step()

                if (
                        cfg.TEST.EVAL_PERIOD > 0
                        and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                        and iteration != max_iter - 1
                ):
                    do_test(cfg, model)
                    # Compared to "train_net.py", the test results are not dumped to EventStorage
                    comm.synchronize()

                if iteration - start_iter > 5 and (
                        (iteration + 1) % 20 == 0 or iteration == max_iter - 1
                ):
                    for writer in writers:
                        writer.write()
                periodic_checkpointer.step(iteration)

                if iteration > prev_iter:
                    prev_iter = iteration
                    if losses_reduced < BEST_LOSS:

                        BEST_LOSS = losses_reduced
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter % 100 == 0:
                            print(f"Loss has not improved for {patience_counter} iterations")
                        if patience_counter >= cfg.PATIENCE:
                            print(f"EARLY STOPPING")
                            periodic_checkpointer.save("model_final")
                            break

    do_test(cfg, model)



if __name__ == "__main__":

    # get path to dataset, if not default
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='path to dataset')
    args = parser.parse_args()

    logger = logging.getLogger("detectron2")
    register_data_sets(display_random_train_data=False,individual_dataset_path=args.data)
    cfg = prepare_cfg()
    download_pretrained_model()
    training=True
    if training:
        logger.info('Start training')
        run_training(cfg,True)
    else:
        run_training(cfg, False)
        logger.info('No training required')
        # store pretrained model as final model if not already existing
        if not os.path.exists('output_low_res/X152/All_X152/model_final.pth'):
            logger.info('No final model found... Copying pretrained model to final model! Otherwise consider training first')
            if os.path.exists('output_low_res'):
                logger.info('store existing output in tar archive, if still needed. Otherwise remove it! Removing existing directory for new creation!')
                with tarfile.open('output_old.tar', "w:gz") as tar:
                    tar.add('output')
                shutil.rmtree('output_low_res')

            os.makedirs('output_low_res')
            os.makedirs('output_low_res/X152')
            os.makedirs('output_low_res/X152/All_X152')

            shutil.copy('pretrained_model/TableBank_X152.pth', 'output_low_res/X152/All_X152/model_final.pth')
        else:
            logger.info('Final model found. Nothing to do...')
