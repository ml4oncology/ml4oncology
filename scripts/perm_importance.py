"""
========================================================================
© 2018 Institute for Clinical Evaluative Sciences. All rights reserved.

TERMS OF USE:
##Not for distribution.## This code and data is provided to the user solely for its own non-commercial use by individuals and/or not-for-profit corporations. User shall not distribute without express written permission from the Institute for Clinical Evaluative Sciences.

##Not-for-profit.## This code and data may not be used in connection with profit generating activities.

##No liability.## The Institute for Clinical Evaluative Sciences makes no warranty or representation regarding the fitness, quality or reliability of this code and data.

##No Support.## The Institute for Clinical Evaluative Sciences will not provide any technological, educational or informational support in connection with the use of this code and data.

##Warning.## By receiving this code and data, user accepts these terms, and uses the code and data, solely at its own risk.
========================================================================
"""
from functools import partial 
import argparse
import pickle
import os
import sys
sys.path.append(os.getcwd())

from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier as placeholderClassifier
pd.options.mode.chained_assignment = None
import numpy as np
import pandas as pd

from src.utility import (load_ml_model, load_ensemble_weights)
from src.config import (
    root_path, cyto_folder, acu_folder, can_folder, death_folder, 
    split_date, blood_types, variable_groupings_by_keyword
)
from src.prep_data import (PrepDataCYTO, PrepDataEDHD, PrepDataCAN)
from src.train import (TrainML, TrainRNN)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', datefmt='%I:%M:%S')

class PermImportance:    
    def __init__(self, output_path, split_date=None, preload=True):
        """
        Args:
            split_date (str): date to temoporally split the data into 
                development and testing cohort. If None, data will be split 
                across all years evenly into train-val-test split
        """
        self.output_path = output_path
        self.split_date = split_date
        
        # get test data with 
        #   1. original features 
        #   2. one-hot encoded features
        orig_data_splits, data_splits = self.get_data()
        _, _, self.X_test, _, _, self.Y_test, = data_splits
        _, _, self.orig_X_test, _, _, _ = orig_data_splits
        
        # initialize RNN and ML Training class - you need some of their member functions
        self.train_rnn = TrainRNN(data_splits, output_path)
        self.train_ml = TrainML(data_splits, output_path)
        
        self.ikns = self.X_test.pop('ikn') # patient id column
        self.dummy_cols = self.X_test.columns
        self.target_events = self.Y_test.columns.tolist()
        self.params =  {'n_repeats': 5, 'random_state': 42}
        
        if preload:
            # load pretrained models 
            self.ensemble_weights = load_ensemble_weights(save_dir=f'{self.output_path}/best_params')
            self.models = {}
            for algorithm in self.ensemble_weights:
                if algorithm == 'RNN':
                    filename = f'{self.output_path}/best_params/RNN_best_param.pkl'
                    with open(filename, 'rb') as file: 
                        rnn_model_param = pickle.load(file)
                    del rnn_model_param['learning_rate']
                    self.models[algorithm] = self.train_rnn.get_model(load_saved_weights=True, **rnn_model_param)
                else:
                    self.models[algorithm] = load_ml_model(self.output_path, algorithm)
        
    def load_data(self):
        raise NotImplementedError
        
    def clean_y(self, Y):
        """You can overwrite this to do custom operations
        """
        return Y
        
    def get_data_splits(self, data, include_ikn=False):
        kwargs = {'target_keyword': self.target_keyword, 'split_date': self.split_date}
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test = self.prep.split_data(data, **kwargs)
        
        if include_ikn:
            # the ikn corresponding to each data split are assigned by matching the indices
            X_train['ikn'], X_valid['ikn'], X_test['ikn'] = data['ikn'], data['ikn'], data['ikn']
        
        data_splits = (
            X_train, X_valid, X_test, 
            self.clean_y(Y_train), self.clean_y(Y_valid), self.clean_y(Y_test)
        ) 
        return data_splits
        
    def get_data(self):
        data = self.load_data()
        
        # original feature data splits
        orig_data_splits = self.get_data_splits(data)
        
        # one-hot encoded feature data splits
        data_splits = self.get_data_splits(self.prep.dummify_data(data), include_ikn=True)

        return orig_data_splits, data_splits
    
    def predict(self, algorithm, model, X, split='Test'):
        X = X.astype(float)
        
        if algorithm == 'ENS':
            pred_prob, ensemble_weights = [], []
            for alg, weight in self.ensemble_weights.items():
                pred_prob.append(self.predict(alg, self.models[alg], X))
                ensemble_weights.append(weight)
            pred_prob = np.average(pred_prob, axis=0, weights=ensemble_weights)
            
        elif algorithm == 'RNN':
            X['ikn'] = self.ikns
            self.train_rnn.dataset_splits[split] = self.train_rnn.transform_to_tensor_dataset(X, self.Y_test)
            pred_prob, index_arr = self.train_rnn._get_model_predictions(self.models[algorithm], split)
            # double check and clean up
            assert all(self.Y_test.index == index_arr)
            del X['ikn']
            
        else:
            self.train_ml.preds[split][algorithm] = None # claer cache
            self.train_ml.data_splits[split] = (X, self.Y_test) # set the new input
            pred_prob, _ = self.train_ml.predict(model, split, algorithm)
            pred_prob = pred_prob.values
        return pred_prob
    
    def get_feature_importance(self, algorithm, columns=None):
        """Run permutation importance across the original, NOT one-hot-encoded columns/features
        """
        if columns is None: columns = self.orig_X_test.columns
        model = None if algorithm == 'ENS' else self.models[algorithm]
        result = pd.DataFrame()
        
        # get original score
        pred_prob = self.predict(algorithm, model, self.X_test)
        orig_scores = roc_auc_score(self.Y_test, pred_prob, average=None)
            
        for col in columns:
            importances = []
            for i in range(self.params['n_repeats']):
                X = self.orig_X_test.copy()
                X[col] = X[col].sample(frac=1, random_state=self.params['random_state']+i).set_axis(X.index)
                X = self.prep.dummify_data(X)
                X[self.dummy_cols.difference(X.columns)] = 0 # add the missing columns
                X = X[self.dummy_cols]
                pred_prob = self.predict(algorithm, model, X)
                importances.append(orig_scores - roc_auc_score(self.Y_test, pred_prob, average=None))
            result.loc[col, self.target_events] = np.mean(importances, axis=0)
            logging.info(f'Successfully computed perm feature importance scores for {col}')
        
        # save results
        filepath = f'{self.output_path}/perm_importance/{algorithm}_feature_importance.csv'
        result.to_csv(filepath, index_label='index')
        
    def get_group_importance(self, algorithm):
        """Run permutation importance across subgroups of columns/features
        """
        model = None if algorithm == 'ENS' else self.models[algorithm]
        result = pd.DataFrame()
        
        # get original score
        pred_prob = self.predict(algorithm, model, self.X_test)
        orig_scores = roc_auc_score(self.Y_test, pred_prob, average=None)
            
        for group, keyword in variable_groupings_by_keyword.items():
            permute_cols = self.dummy_cols[self.dummy_cols.str.contains(keyword)]
            importances = []
            for i in range(self.params['n_repeats']):
                X = self.X_test.copy()
                X[permute_cols] = X[permute_cols].sample(frac=1, random_state=self.params['random_state']+i).set_index(X.index)
                pred_prob = self.predict(algorithm, model, X)
                importances.append(orig_scores - roc_auc_score(self.Y_test, pred_prob, average=None))
            result.loc[group, self.target_events] = np.mean(importances, axis=0)
            logging.info(f'Successfully computed perm group importance scores for {group}')
            
        # save results
        filepath = f'{self.output_path}/perm_importance/{algorithm}_group_importance.csv'
        result.to_csv(filepath, index_label='index')
            
class PermImportanceCYTO(PermImportance):
    """Permutation importance for cytopenia
    """
    def __init__(self, output_path):
        self.prep = PrepDataCYTO()
        self.target_keyword = 'target'
        super().__init__(output_path, split_date=split_date)
        
    def load_data(self):
        return self.prep.get_data(missing_thresh=75)
    
class PermImportancePROACCT(PermImportance):
    """Permutation importance for acute care use (ED/H)
    ED - Emergency Department visits
    H - Hospitalizations
    """
    def __init__(self, output_path, days=30):
        self.target_keyword = f'_within_{days}days'
        self.prep = PrepDataEDHD('acu')
        super().__init__(output_path)
    
    def load_data(self):
        return self.prep.get_data(self.target_keyword, missing_thresh=80)
    
    def clean_y(self, Y):
        Y.columns = Y.columns.str.replace(self.target_keyword, '')
        return Y
    
class PermImportanceDEATH(PermImportance):
    def __init__(self, output_path):
        self.target_keyword = 'Mortality'    
        self.prep = PrepDataEDHD('death')
        super().__init__(output_path, split_date=split_date)
    
    def load_data(self):
        return self.prep.get_data(self.target_keyword, missing_thresh=75, treatment_intents=['P'])
    
class PermImportanceCAN(PermImportance):
    """Permutation importance for cisplatin-induced nephrotoxicity
    """
    def __init__(self, output_path, adverse_event):
        self.prep = PrepDataCAN(adverse_event=adverse_event)
        self.target_keyword = 'SCr|dialysis|next'
        super().__init__(output_path, split_date=split_date)
        
    def load_data(self):
        return self.prep.get_data(missing_thresh=80)

def main(pm, algorithm='ENS', permute_group=False):
    if permute_group:
        pm.get_group_importance(algorithm=algorithm)
    else:
        pm.get_feature_importance(algorithm=algorithm)
        
def main_cyto(permute_group=False):
    output_path = f'{root_path}/{cyto_folder}/models'
    pm = PermImportanceCYTO(output_path)
    main(pm, permute_group=permute_group)

def main_acu(days=30, permute_group=False):
    output_path = f'{root_path}/{acu_folder}/models/within_{days}_days'
    pm = PermImportancePROACCT(output_path, days=days)
    main(pm, permute_group=permute_group)
    
def main_death(permute_group=False):
    output_path = f'{root_path}/{death_folder}/models'
    pm = PermImportanceDEATH(output_path)
    main(pm, permute_group=permute_group)
        
def main_can(adverse_event='aki', permute_group=False):
    output_path = f'{root_path}/{can_folder}/models/{adverse_event.upper()}'
    pm = PermImportanceCAN(output_path, adverse_event)
    main(pm, permute_group=permute_group)
    
if __name__ == '__main__':
    main_events = {
        'CYTO': main_cyto, # Cytopenia
        'ACU': main_acu,  # Acute Care Use
        'CAAKI': partial(main_can, adverse_event='aki'), # Cisplatin-Associated Nephrotoxicicity
        'CACKD': partial(main_can, adverse_event='ckd'), # Cisplatin-Associated Chronic Kidney Disease
        'DEATH': main_death
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--adverse-event', default='CYTO', choices=main_events.keys())
    parser.add_argument(
        '--permute-group', action='store_true', default=False, 
        help=("Run permutation group importance (permute subset of alike features) "
              "instead of permutation feature importance (permute each feature)")
    )
    args = vars(parser.parse_args())
    main_event = main_events[args['adverse_event']]
    main_event(permute_group=args['permute_group'])
