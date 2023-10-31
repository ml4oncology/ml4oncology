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
from collections import defaultdict

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from skmisc.loess import loess
from tqdm import tqdm
import numpy as np
import pandas as pd

from src.conf_int import bootstrap_sample
from src.config import blood_types, clean_variable_mapping

###############################################################################
# Baseline Models
###############################################################################
class BaselineModel:
    """Baseline model in which a single predictor/variable/column is used to 
    predict a target
    """
    def __init__(self, **params):
        self.params = params
        self.models = {}
        self.target_events = []
        
    def _fit(self, x, y):
        raise NotImplementedError
        
    def _predict(self, model, x):
        raise NotImplementedError
        
    def _predict_with_confidence_interval(self, target_event, x, **kwargs):
        raise NotImplementedError
        
    def fit(self, x, Y):
        for target_event, y in Y.items():
            self.models[target_event] = self._fit(x, y)
        self.target_events = Y.columns.tolist()
        
    def predict(self, x):
        preds = {target_event: self._predict(self.models[target_event], x)
                 for target_event in self.target_events}
        return preds
    
    def predict_with_confidence_interval(self, x, **kwargs):
        preds, ci_lower, ci_upper = {}, {}, {}
        for te in self.target_events:
            output = self._predict_with_confidence_interval(te, x, **kwargs)
            preds[te], ci_lower[te], ci_upper[te] = output
        return preds, ci_lower, ci_upper
    
class LOESSModel(BaselineModel):
    """Locally estimated scatterplot smoothing (LOESS) model
    """
    def _fit(self, x, y):
        model = loess(x, y, **self.params)
        model.fit()
        return model

    def _predict(self, model, x):
        return model.predict(x).values
    
    def _predict_with_confidence_interval(self, target_event, x, **kwargs):
        model = self.models[target_event]
        pred = model.predict(x, stderror=True)
        ci = pred.confidence(0.05)
        return pred.values, ci.lower, ci.upper
    
class PolynomialModel(BaselineModel):
    """Polynomial Model
    
    Supported algorithms include:
    - SPLINE: B-spline basis functions (piecewise polynomials)
    - POLY: pure polynomials
    """
    def __init__(self, alg, task_type, **params):
        """
        Args:
            task_type (str): the type of machine learning task, either `C` for 
                classifcation or `R` for regression
        """
        super().__init__(**params)
        poly_models = {'SPLINE': SplineTransformer, 'POLY': PolynomialFeatures}
        reg_models = {'R': Ridge, 'C': LogisticRegression}
        self.poly_model = poly_models[alg]
        self.reg_model = reg_models[task_type]
        self.task_type = task_type
        
    def _format(self, x):
        if len(x.shape) == 1: x = np.expand_dims(x, axis=1)
        return x
        
    def _fit(self, x, y):
        model = make_pipeline(
            self.poly_model(**self.params['PLY']), 
            self.reg_model(**self.params['REG'])
        )
        model.fit(self._format(x), y)
        return model
    
    def _predict(self, model, x):
        if self.task_type == 'R':
            return model.predict(self._format(x))
        elif self.task_type == 'C':
            return model.predict_proba(self._format(x))[:, 1]
        else:
            raise ValueError(f'Unknown task type: {self.task_type}')
    
    def _compute_confidence_interval(self, x, x_train, y_train, n_bootstraps=1000):
        ci_preds = []
        for random_seed in tqdm(range(n_bootstraps)):
            Y = bootstrap_sample(y_train, random_seed)
            X = x_train.loc[Y.index]
            model = self._fit(X, Y)
            pred = self._predict(model, x)
            ci_preds.append(pred)
        return np.array(ci_preds)
    
    def _predict_with_confidence_interval(
        self, 
        target_event, 
        x, 
        x_train=None,
        y_train=None
    ):
        """For spline and poly algorithm, confidence interval is calculated by 
        permuting the data used to originally train the model, retraining the 
        model, recomputing the prediction for x, and taking the 2.5% and 97.5% 
        quantiles of each prediction for sample n in x
        """
        model = self.models[target_event]
        pred =  self._predict(model, x)
        ci_preds = self._compute_confidence_interval(
            x, x_train, y_train[target_event]
        )
        lower, upper = np.percentile(ci_preds, [2.5, 97.5], axis=0).round(3)
        return pred, lower, upper

###############################################################################
# Simple Baseline Models
###############################################################################
class SimpleBaselineModel:
    """This baseline model outputs the corresponding 
    target rate of
    1. each category of a categorical column
    2. each bin of a numerical column
    from the training set. 

    E.g. if patient is taking regimen X, baseline model outputs target rate 
         of regimen X in the training set
    E.g. if patient's blood count measurement is X, baseline model outputs 
         target rate of blood count bin in which X belongs to

    Each column acts as a baseline model.

    NOTE: Why not predict previously measured blood count directly? 
    Because we need prediction probability to calculate AUROC.
    Predicting if previous blood count is less than x threshold will output 
    a 0 or 1, resulting in a single point on the ROC curve.
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def predict(self, target_events=None, splits=None):
        if target_events is None: target_events = list(self.labels['Test'].columns)
        if splits is None: splits = ['Valid', 'Test']
        var_mapping = {
            'baseline_eGFR': 'Gloemrular Filteration Rate', 
            **{f'baseline_{bt}_count': 'Blood Count' for bt in blood_types},
            **clean_variable_mapping
        }
        
        preds = defaultdict(dict)
        for base_col, base_vals in self.data.items():
            mean_targets = defaultdict(dict)
            Y = self.labels['Train']
            X = base_vals[Y.index]
            numerical_col = X.dtype == float
            if numerical_col: X, bins = pd.cut(X, bins=100, retbins=True)
            
            # compute target rate of each category or numerical bin of the 
            # column
            for group_name, group in X.groupby(X):
                means = Y.loc[group.index].mean()
                for target_event, mean in means.items():
                    mean_targets[target_event][group_name] = mean
            
            # get baseline algorithm name
            name = var_mapping.get(base_col, base_col)
            if numerical_col: name += ' Bin'
            alg = f'Baseline - Event Rate Per {name}'.replace('_', ' ').title()
            
            # special case for blood count measurements 
            # only use the blood type column matching the blood type target
            bt = base_col.replace('baseline_', '').replace('_value', '')
            if bt in blood_types:
                target_names = [blood_types[bt]['cytopenia_name']] 
            else:
                target_names = target_events
            
            # get predictions for each split
            for split in splits:
                idxs = self.labels[split].index
                X = base_vals[idxs]
                if numerical_col: X = pd.cut(X, bins=bins).astype(object)
                pred = {tn: X.map(mean_targets[tn]).fillna(0) for tn in target_names}
                preds[alg][split] = pd.DataFrame(pred)
        return preds
        