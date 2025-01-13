import os
import cv2
import logging
from PIL import Image
from wand.image import Image as Wand
import torch
import numpy as np
import io
import sys

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.engine import default_setup


def crop_object(image, box, bias):
  """Crops an object from an image

  Inputs:
    image: PIL image
    box: one box from Detectron2 pred_boxes
    bias: bias for which the detected box is increased in each direction
  """

  x_top_left = box[0] - bias
  y_top_left = box[1] - bias
  x_bottom_right = box[2] + bias
  y_bottom_right = box[3] + int(bias/2)

  crop_img = image.crop((int(x_top_left), int(y_top_left), int(x_bottom_right), int(y_bottom_right)))
  return crop_img

def create_cfg(path_to_final_model):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file("../detectron2_segmentation/pretrained_model/TableBank_X152.yaml")
    default_setup(cfg, "detectron2_segmentation/pretrained_model/TableBank_X152.yaml")
    cfg.MODEL.WEIGHTS = path_to_final_model
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    return cfg

def transform_coords(x_val,y_val,low_res_width,high_res_width,low_res_height,high_res_height):
    low_res_x = x_val
    low_res_y = y_val

    high_res_x = low_res_x / low_res_width * high_res_width
    high_res_y = low_res_y / low_res_height * high_res_height

    return [high_res_x,high_res_y]


def main(years = None):
    """Performs the table detection

     Notes:
       - Path to low resolution images needs to be set (needed for detecting tables, since the models are trained with
            low resolution images)
       - Path to high resolution images needs to be set (needed for cropping tables, since high resolution is better
            for further processing)
       - Output dir needs to be set (needed for saving cropped images)
     """

    path_images_low_res = '../data/000_Raw_Images/pictures_all_low_res'
    path_images_high_res = '../data/000_Raw_Images/pictures_all'
    output_dir = '../data/000_Raw_Images/cropped_tables'

    if years is not None:
        years = [years]

    logger = logging.getLogger("detectron2")


    if years is None:
        #create output structure
        if os.path.exists(output_dir):
            print('cropped tables already existing. Check results first and delete them if you want to proceed')
            sys.exit()
        os.makedirs(output_dir)



    if years is None:
        years = os.listdir(path_images_low_res)


    for year in sorted(years):

        if year == '1910':
            path_to_final_model = '../detectron2_segmentation/output_low_res_1910/X152/All_X152/model_final.pth'
        else:
            path_to_final_model = '../detectron2_segmentation/output_low_res/X152/All_X152/model_final.pth'

        cfg = create_cfg(path_to_final_model)
        predictor = DefaultPredictor(cfg)

        if not os.path.exists(os.path.join(output_dir,year)):
            os.mkdir(os.path.join(output_dir,year))

        files = [x for x in os.listdir(os.path.join(path_images_low_res,year)) if not x.startswith('.')]

        for file in files:
            print(file)
            if not os.path.exists(os.path.join(output_dir, year,file.rstrip('.jpg'))):
                os.mkdir(os.path.join(output_dir, year,file.rstrip('.jpg')))

            path_file_low_res = os.path.join(path_images_low_res,year,file)

            # deskew and convert to PIL image
            with Wand(filename=path_file_low_res) as wand_img:
                wand_img.deskew(0.4 * wand_img.quantum_range)

                img_buffer = np.asarray(bytearray(wand_img.make_blob(format='png')), dtype='uint8')
                bytesio = io.BytesIO(img_buffer)
                pil_img = Image.open(bytesio)

            # convert PIL to cv2 image
            opencvImage = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # do prediction to find tables
            outputs = predictor(opencvImage)

            if year == '1910':
                # Get pred_boxes from Detectron2 prediction outputs if score is above 90%
                indices = [idx for idx, val in enumerate(list(outputs["instances"].scores)) if
                           outputs["instances"].scores[idx] > 0.90]
            else:
                # Get pred_boxes from Detectron2 prediction outputs if score is above 80%
                indices = [idx for idx,val in enumerate(list(outputs["instances"].scores)) if outputs["instances"].scores[idx]>0.80]

            for idx in indices:
                box = list(outputs["instances"].pred_boxes[idx])[0].detach().cpu().numpy()

                # img low res deskewed
                image_low_res = pil_img
                low_res_width = image_low_res.size[0]
                low_res_height = image_low_res.size[1]

                # deskew img high res
                with Wand(filename=os.path.join(path_images_high_res,year,file)) as wand_img:
                    wand_img.deskew(0.4 * wand_img.quantum_range)

                    img_buffer = np.asarray(bytearray(wand_img.make_blob(format='png')), dtype='uint8')
                    bytesio = io.BytesIO(img_buffer)
                    image_high_res = Image.open(bytesio)
                high_res_width = image_high_res.size[0]
                high_res_height = image_high_res.size[1]

                # transform coordinates to match larger picture
                new_points = []
                new_points.append(transform_coords(box[0],box[1],low_res_width,high_res_width,low_res_height,high_res_height))
                new_points.append(transform_coords(box[2], box[3], low_res_width, high_res_width, low_res_height, high_res_height))
                flat_list = [x for xs in new_points for x in xs]

                # the big tables of 1910 need a small bias
                if year == '1910':
                    crop_img = crop_object(image_high_res, flat_list,bias=60)
                else:
                    crop_img = crop_object(image_high_res, flat_list, bias=00)

                print('Process: '+os.path.join(year,file))
                crop_img.save(os.path.join(output_dir, year,file.rstrip('.jpg'),'table_'+str(idx)+'.jpg'))

                os.makedirs(os.path.join(output_dir,'scores', year,file.rstrip('.jpg')),exist_ok=True)
                with open(os.path.join(output_dir,'scores', year,file.rstrip('.jpg'),"scores.txt"), "a") as f:
                    f.write("Score for table_"+str(idx)+': '+' '+str(outputs["instances"].scores[idx].cpu().item())+'\n')


if __name__ == "__main__":
    main()
