import json
import os
import re
import sklearn.metrics
import numpy as np
import Levenshtein

def structure_evaluate(truth_list, prediction_list,enable_output = True):
    """
    Evaluation of the automatic structuring of the OCR results.
    

    Parameters
    ----------
    prediction_list : list of dict
        List of structured information from the OCRd table rows
    truth_list : list of dict
        List of manually structured information
    
    Returns
    ----------
    tuple of f1_score and classification report

    Notes
    ----------
    To allow the evaluation, two precondition must hold
    1. Both input lists have to be of the same length.
    2. Each dict must contain the same keys. For keys, where values are not available, None must be used.
    """
    assert len(prediction_list) == len(truth_list), "Prediction and truth differ in length"

    def split_string(s:str):
        if not s:
            return []
        return [e for e in re.split('[ ,|]', s) if e != '']

    results = []


    for pred_entry, truth_entry in zip(prediction_list, truth_list):
        assert pred_entry.keys() == truth_entry.keys(), "Keys in prediction and truth differ"
        
        pred = {k:split_string(v) for k,v in pred_entry.items()}
        truth = {k:split_string(v) for k,v in truth_entry.items()}
        for k,v in truth.items():
            for word in v:
                word_found = False
                # search key in prediction with this word
                for pk,pv in pred.items():
                    if word in pv:
                        results.append((k, pk))
                        word_found = True
                        break
                if not word_found:
                    results.append((k,"None"))
                    if enable_output:
                        print(f"Word {word} in key {k} from truth was not found in prediction")

    prediction_classes, truth_classes = zip(*results)
    
    report = sklearn.metrics.classification_report(truth_classes, prediction_classes)
    f1_score = sklearn.metrics.f1_score(truth_classes, prediction_classes, average='micro')

    return f1_score, report

def evaluate_OCR(ocr_data, ground_truth_data_string):
    """
        Evaluation of the OCR results.


        Parameters
        ----------
        ocr_data : ocr data as a list of lines
        ground_truth_data_string : ground truth data as a list of lines

        Returns
        ----------
        eval_string : evaluation string containing:
            - a direct comparison of ground truth and OCR result
            - the Levenshtein distance and the caracter error rate (CER) for each row
            - the final CER and a mean CER
        """

    sum_d = 0
    sum_len = 0
    cer_list = []
    eval_string = []

    # compare corrected output
    for pred, truth in list(zip(np.array(ocr_data), np.array(ground_truth_data_string))):
        pred = re.sub(r'\s+', ' ', pred)
        d = Levenshtein.distance(truth, pred)
        if len(truth) != 0:

            cer = d / len(truth)
            cer_list.append(cer)
            sum_d += d
            sum_len += len(truth)
            eval_string.append(f" OCR:\t{pred} \n Truth:\t{truth} \n Levenshtein: {d}\tCER: {cer: .3}\n")
        else:
            eval_string.append(f" OCR:\t{pred} \n Truth:\t{truth} \n cannot calculate Levenshtein and CER \n -> does not influence result\n" )
    eval_string.append(f"final CER: {sum_d / sum_len} mean CER: {np.mean(cer_list)}")
    return eval_string