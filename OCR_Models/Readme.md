# OCR-Models
This directory is used for the used OCR models, which can be downloaded as described below:
- [Fraktur.traineddata](https://github.com/tesseract-ocr/tessdata_best/raw/refs/heads/main/script/Fraktur.traineddata):
        An LSTM model specifically trained for recognizing Fraktur in documents from the 19th and early 20th centuries.
        It is commonly used for historical texts but shows weaknesses with modern fonts or heavily damaged documents.
- [Fraktur_GT4HistOCR.traineddata](https://ub-backup.bib.uni-mannheim.de/~stweil/ocrd-train/data/GT4HistOCR/tessdata_best/GT4HistOCR.traineddata):
        Developed as part of the "Ground Truth for Training OCR for Historical Documents" (GT4HistOCR) project.
        Trained on a comprehensive dataset specifically optimized for recognizing historical Fraktur.
        It performs well on documents with highly varied fonts and qualities, making it suitable for historical documents.
- [frak2021.traineddata](https://ub-backup.bib.uni-mannheim.de/~stweil/tesstrain/frak2021/tessdata_best/frak2021-0.905.traineddata):
        An advanced LSTM model that was trained on the GT4HistOCR dataset as well as the AustrianNewspapers and the Fibeln datasets.
        Often delivers better results for texts with additional linguistic variations and a wider range of historical fonts.