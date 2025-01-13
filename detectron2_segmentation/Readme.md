# Detectron2
This directory contains the required Detectron2 logic for training and usage:
- pretrained_model
  - contains information for the pretrained TableBank_152 model. The TableBank_152.pth file must be placed here for
    further fine-tuning
- config.py
  - conains some basic information
- prepare_dataset.py
  - splits the dataset in train, test and validation data and prepares it for training
- rcnn_model.py
  - contains model structure
- train_detectron2.py
  - contains the training routine