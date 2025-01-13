# Evaluation

This directory contains the main implementation of the pipeline evaluation. Naturally, the evaluation can only be done 
on the annotated ocr dataset:
- evaluate_pipeline.ipynb
  - notebook that contains the evaluation of the complete pipeline showing all results
- prepare_ocr_dataset.py
  - script that divides the dataset into train, test and validation data
- evaluate_ocr.py
  - script that evaluates the ocr results of the three models Fraktur, GT4HistOCR and frak2021 by comparing it to the GT
- OCR_correction_for_evaluation.py
  - script that performs the OCR-correction only for data for which a GT is available. The correction is done for 
    different model orders given to the LLM prompt
- evaluate_corrected_ocr.py
  - script that evaluates the corrected ocr results by comparing it to the GT
- structuring_for_evaluation.py
  - script that performs the structuring only for data for which a GT is available. The structuring can be done on the GT
    alone or on the OCR corrected data
- evaluate_structuring.py
  - script that evaluates the structuring performance by comparing it to the GT