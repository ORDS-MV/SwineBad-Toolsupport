# Data
This directory contains the data
- 000_Raw_Images
  - contains the raw images in two sub-directories 'pictures_all_low_res' and 'pictures_all'. 
  - contains the cropped tables
- 001_Annotation
  - contains the OCR data annotation
- 002_OCR_all_tables
  - contains the OCR results of the three different models Fraktur,GT4HistOCR and frak2021 of all cropped tables
- 003_OCR_evaluation
  - contains the results of the OCR-evaluation process
- 004_OCR_correction_all_tables
  - contains the corrected OCR results of all OCR data given in 002_OCR_all_tables
- 004_OCR_correction_evaluation
  - contains the results of the evaluation of the ocr correction
- 005_Structuring_all_tables
  - contains the structuring results of all corrected OCR files given in 004_OCR_correction_all_tables
- 005_Structuring_evaluation
  - contains the results of the evaluation of the structuring process