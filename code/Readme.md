# Code
This directory contains the main implementation of the pipeline:
- 00_download_pictures_from_library.py
  - downloads the pictures from the digital library
- 01_crop_segmentation.py
  - script to apply the table detection and cropping operation
- 01_crop_segmentation_multiprocessing.py
  - script to apply the table detection and cropping operation using multiple processes in parallel
- 02_OCR_tesseract.py
  - script to do the ocr extraction with three models
- 02_do_OCR_multiprocessing.py
  - script to do the ocr extraction with three models using multiple processes in parallel
- 03_OCR_correction.py
  - script to do the ocr correction using an LLM
- 04_structuring.py
  - script to do the structuring using an LLM