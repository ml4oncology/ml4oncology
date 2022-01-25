import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import tqdm
import pickle
from functools import partial

from scripts.utilities import (load_ml_model, load_ensemble_weights)
from scripts.config import (root_path, blood_types, cytopenia_thresholds, ml_models)
from scripts.prep_data import (PrepData, PrepDataEDHD)

from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier as placeholderClassifier

class PermImportance:
    """
    Run permuation feature importance
    """        
    def __init__(self):
        self.output_path = None
        self.prep = None
        self.dummy_cols = None
        self.target_keyword = None
        self.target_types = None
        self.ensemble_weights = None
        raise NotImplementedError
        
    def get_model_data(self):
        raise NotImplementedError
        
    def clean_y_valid(self):
        raise NotImplementedError
        
    def get_data(self):
        model_data = self.get_model_data()
        model_data, clip_thresholds = self.prep.clip_outliers(model_data)
        dummy_cols = self.prep.dummify_data(model_data).columns
        target_cols = dummy_cols[dummy_cols.str.contains(self.target_keyword)]
        dummy_cols = dummy_cols.drop(target_cols.tolist() + ['ikn'])
        _, (X_valid, Y_valid), _ = self.prep.split_data(model_data, target_keyword=self.target_keyword, convert_to_float=False)
        Y_valid = self.clean_y_valid(Y_valid)
        
        return X_valid, Y_valid, dummy_cols
    
    def predict(self, idx, algorithm, model, X):
        if algorithm == 'NN':
            pred_prob = model.predict_proba(X)
            pred_prob = pred_prob.T[idx]
        else:
            pred_prob = model.estimators_[idx].predict_proba(X)
            pred_prob = pred_prob[:, 1]
        return pred_prob
    
    def scorer(self, model, X, Y, target_type='ACU', algorithm='ENS'):
        X = self.prep.dummify_data(X)
        # add the missing columns
        X[self.dummy_cols.difference(X.columns)] = 0
        X = X[self.dummy_cols]
        X = X.astype(float)
        
        idx = self.target_types.index(target_type)
        if algorithm == 'ENS':
            pred_prob = []
            for algorithm in ml_models:
                model = load_ml_model(self.output_path, algorithm)
                pred_prob.append(self.predict(idx, algorithm, model, X))
            pred_prob = np.average(pred_prob, axis=0, weights=self.ensemble_weights)
        else:
            pred_prob = self.predict(idx, algorithm, model, X)
            
        return roc_auc_score(Y, pred_prob)
    
    def run(self, algorithm):
        model = placeholderClassifier() if algorithm == 'ENS' else load_ml_model(self.output_path, algorithm)
        result = pd.DataFrame(index=self.X_valid.columns)
        params = {'n_jobs': 32, 'n_repeats': 15, 'random_state': 42}
        for target_type in self.target_types:
            scorer = partial(self.scorer, target_type=target_type, algorithm=algorithm)
            feature_importances = permutation_importance(model, self.X_valid, self.Y_valid[target_type], 
                                                         scoring=scorer, **params)
            result[target_type] = feature_importances.importances_mean
            print(f'Successfully computed perm importance scores for {target_type}')
        result.to_csv(f'{self.output_path}/perm_importance/{algorithm}.csv', index_label='index')
            
class PermImportanceCyto(PermImportance):
    def __init__(self, output_path):
        self.prep = PrepData()
        self.output_path = output_path
        self.target_keyword = 'target'
        self.X_valid, self.Y_valid, self.dummy_cols = self.get_data()
        self.target_types = self.Y_valid.columns.tolist()
        
    def get_model_data(self):
        return pd.read_csv(f'{self.output_path}/model_data.csv', dtype={'curr_morph_cd': str, 'lhin_cd': str})
    
    def clean_y_valid(self, Y_valid):
        return self.prep.regression_to_classification(Y_valid, cytopenia_thresholds)
    
class PermImportanceEDHD(PermImportance):
    def __init__(self, output_path, days):
        self.prep = PrepDataEDHD()
        self.output_path = output_path
        self.target_keyword = f'_within_{days}days'
        self.X_valid, self.Y_valid, self.dummy_cols = self.get_data()
        self.target_types = self.Y_valid.columns.tolist()
        self.ensemble_weights = load_ensemble_weights(save_dir=f'{output_path}/best_params', ml_models=ml_models)
    
    def get_model_data(self):
        return self.prep.get_data(f'{root_path}/ED-H-D', self.target_keyword)
    
    def clean_y_valid(self, Y_valid):
        Y_valid.columns = Y_valid.columns.str.replace(self.target_keyword, '')
        return Y_valid
        
def cyto_main():
    output_path = f'{root_path}/cytopenia/models'
    pm = PermImportanceCyto(output_path)
    for alg in ml_models + ['ENS']:
        pm.run(alg)

def edhd_main():
    days = 30
    output_path = f'{root_path}/ED-H-D/models/ML/within_{days}_days'
    pm = PermImportanceEDHD(output_path, days)
    for alg in ml_models + ['ENS']:
        if alg in ['LR', 'XGB', 'RF', 'NN']: continue
        print(f'Running permutation importance for {alg} model')
        pm.run(alg)
    
if __name__ == '__main__':
    edhd_main()


