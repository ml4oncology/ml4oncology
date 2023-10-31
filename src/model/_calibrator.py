"""
========================================================================
© 2023 Institute for Clinical Evaluative Sciences. All rights reserved.

TERMS OF USE:
##Not for distribution.## This code and data is provided to the user solely for its own non-commercial use by individuals and/or not-for-profit corporations. User shall not distribute without express written permission from the Institute for Clinical Evaluative Sciences.

##Not-for-profit.## This code and data may not be used in connection with profit generating activities.

##No liability.## The Institute for Clinical Evaluative Sciences makes no warranty or representation regarding the fitness, quality or reliability of this code and data.

##No Support.## The Institute for Clinical Evaluative Sciences will not provide any technological, educational or informational support in connection with the use of this code and data.

##Warning.## By receiving this code and data, user accepts these terms, and uses the code and data, solely at its own risk.
========================================================================
"""
from sklearn.isotonic import IsotonicRegression
import numpy as np
import pandas as pd

from src.utility import load_pickle, save_pickle

class IsotonicCalibrator:
    """Calibrator. Matches predicted probability with empirical probability via 
    isotonic regression
    """
    def __init__(self, target_events):
        self.target_events = target_events
        self.regressor = {target_event: IsotonicRegression(out_of_bounds='clip')
                          for target_event in self.target_events}
    
    def load_model(self, model_dir, alg):
        self.regressor = load_pickle(model_dir, f'{alg}_calibrator')
        assert all(target in self.regressor for target in self.target_events)
        
    def save_model(self, model_dir, alg):
        save_pickle(self.regressor, model_dir, f'{alg}_calibrator')
        
    def calibrate(self, pred, label):
        # Reference: github.com/scikit-learn/scikit-learn/blob/main/sklearn/calibration.py
        for target_event in self.target_events:
            self.regressor[target_event].fit(
                pred[target_event], label[target_event]
            )
        
    def predict(self, pred_prob):
        result = []
        for target_event, predictions in pred_prob.items():
            try:
                result.append(self.regressor[target_event].predict(predictions))
            except AttributeError as e:
                if str(e) == "'IsotonicRegression' object has no attribute 'X_min_'":
                    raise ValueError('Model has not been calibrated. '
                                     'Please calibrate the model first')
                else: 
                    raise e
        result = np.array(result).T
        result = pd.DataFrame(
            result, 
            columns=pred_prob.columns, 
            index=pred_prob.index
        )
        return result