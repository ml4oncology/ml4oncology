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
import argparse
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
np.random.seed(42)
import shap

from src import logger
from src.config import split_date
from src.prep_data import PrepDataEDHD
from src.train import TrainRNN
from src.utility import load_pickle, save_pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adverse-event', type=str, required=True, choices=['ACU', 'AKI', 'CKD', 'CYTO', 'DEATH'])
    parser.add_argument('--output-path', type=str, default='./')
    parser.add_argument('--algorithm', type=str, default='ENS', choices=['ENS', 'LR', 'XGB', 'RF', 'NN', 'RNN'])
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    adverse_event = args.adverse_event
    output_path = args.output_path
    algorithm = args.algorithm
    
    if adverse_event != 'DEATH':
        raise NotImplementedError('Sorry, only adverse_event DEATH is currently supported')
    if algorithm != 'ENS':
        raise NotImplementedError('Sorry, only algorithm ENS is currently supported')
    
    # set up the data
    prep = PrepDataEDHD(adverse_event='death', target_keyword='Mortality')
    model_data = prep.get_data(missing_thresh=80, treatment_intents=['P'], verbose=False)
    X, Y, tag = prep.split_and_transform_data(
        model_data, split_date=split_date, remove_immediate_events=True, ohe_kwargs={'verbose': False}
    )
    model_data = model_data.loc[tag.index] # remove sessions in model_data that were excluded during split_and_transform
    
    test_mask = tag['split'] == 'Test'
    x = pd.concat([
        X[test_mask].astype(float), 
        Y[test_mask].astype(float), 
        tag.loc[test_mask, 'ikn']
    ], axis=1)
    sampled_ikns = np.random.choice(x['ikn'].unique(), size=1000)
    mask = x['ikn'].isin(sampled_ikns)
    x, bg_dist = x[mask], x[~mask]
    
    # load models
    lr_model = load_pickle(output_path, 'LR')
    rf_model = load_pickle(output_path, 'RF')
    xgb_model = load_pickle(output_path, 'XGB')
    nn_model = load_pickle(output_path, 'NN')
    ensemble_weights = load_pickle(f'{output_path}/best_params', 'ENS_params')
    rnn_param = load_pickle(f'{self.output_path}/best_params', 'RNN_params')
    trainer = Trainer(X, Y, tag, output_path)
    rnn_model = trainer.models.get_model('RNN', trainer.n_features, trainer.n_targets, **rnn_param)
    rnn_model.load_weights(f'{self.output_path}/{rnn_param["model"]}')
    
    # predict function
    idx = Y.columns.tolist().index('365d Mortality')
    drop_cols = ['ikn'] + Y.columns.tolist()
    def _predict(alg, X):
        if alg == 'RNN':
            trainer.ikns = X['ikn']
            trainer.labels['Test'] = X[trainer.target_events] # dummy variable
            X = X.drop(columns=drop_cols)
            trainer.datasets['Test'] = X # set the new input
            pred = trainer.predict(model, 'Test', alg, calibrated=True)
            return pred.to_numpy()[:, idx]

        X = X.drop(columns=drop_cols)
        if alg == 'LR':
            return lr_model.estimators_[idx].predict_proba(X)[:, 1]
        elif alg == 'XGB':
            return xgb_model.estimators_[idx].predict_proba(X)[:, 1]
        elif alg == 'RF':
            return rf_model.estimators_[idx].predict_proba(X)[:, 1]
        elif alg == 'NN':
            return nn_model.predict_proba(X)[:, idx]
        else:
            raise ValueError(f'{alg} not supported')

    def predict(X):
        weights, preds = [], []
        for alg, weight in ensemble_weights.items():
            weights.append(weight)
            preds.append(_predict(alg, X))
        pred = np.average(preds, axis=0, weights=weights)
        return pred
    
    # compute shap values for the ENS model
    explainer = shap.Explainer(predict, bg_dist)
    shap_values = explainer(x, max_evals=800)
    save_pickle(shap_values, f'{output_path}/feat_importance', 'shap_values_1000_patient')
    
if __name__ == '__main__':
    main()
