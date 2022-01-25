import sys
import os
sys.path.append(os.getcwd())
import tqdm
import pandas as pd
import numpy as np
import multiprocessing as mp
import pickle

from sklearn.multioutput import MultiOutputClassifier
from sklearn.calibration import CalibratedClassifierCV

from scripts.utilities import (load_predictions, subgroup_performance_summary)
from scripts.config import (root_path, event_map, calib_param_logistic)
from scripts.prep_data import (PrepDataEDHD)
from scripts.train import (Train, TrainGRU)

class TrainEDHD(Train):
    def __init__(self, dataset, clip_thresholds=None, n_jobs=32):
        super(TrainEDHD, self).__init__(dataset, clip_thresholds, n_jobs)
        
    def get_LR_model(self, C, max_iter=100):
        params = {'C': C, 
                  'class_weight': 'balanced',
                  'max_iter': max_iter,
                  'random_state': 42, 
                  'solver': 'sag',
                  'tol': 1e-3}
        model = MultiOutputClassifier(CalibratedClassifierCV(
                                        self.ml_models['LR'](**params), 
                                        n_jobs=9,
                                        **calib_param_logistic), 
                                      n_jobs=9)
        return model
    
class PrepDataGRU(PrepDataEDHD):   
    def clean_feature_target_cols(self, feature_cols, target_cols):
        return feature_cols, target_cols
    def extra_norm_cols(self):
        return super().extra_norm_cols() + ['days_since_true_prev_chemo']
        
def get_train(main_dir, output_path, days=30, gru=False, processes=64):
    target_keyword = f'_within_{days}days'
    prep = PrepDataGRU() if gru else PrepDataEDHD()
    model_data = prep.get_data(main_dir, target_keyword, rem_days_since_prev_chemo=not gru)
    model_data, clip_thresholds = prep.clip_outliers(model_data, lower_percentile=0.001, upper_percentile=0.999)
    model_data = prep.dummify_data(model_data)
    train, valid, test = prep.split_data(model_data, target_keyword=target_keyword, convert_to_float=False)
    
    X_train, Y_train = train
    X_valid, Y_valid = valid
    X_test, Y_test = test
    
    Y_train.columns = Y_train.columns.str.replace(target_keyword, '')
    Y_valid.columns = Y_valid.columns.str.replace(target_keyword, '')
    Y_test.columns = Y_test.columns.str.replace(target_keyword, '')
        
    # Initialize and Return Training class
    dataset = (X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
    return TrainGRU(dataset, output_path) if gru and isinstance(gru, bool) else TrainEDHD(dataset, n_jobs=processes)

def main_ci(days=30, subgroup_ci=False, split_ci=False):
    main_dir = f'{root_path}/ED-H-D'
    output_path = f'{main_dir}/models/ML/within_{days}_days'
    train = get_train(main_dir, output_path, days=days)
    if split_ci:
        # get ensemble weights
        filename = f'{output_path}/best_params/ENS_classifier_best_param.pkl'
        with open(filename, 'rb') as file:
            ensemble_weights = pickle.load(file)
        ensemble_weights = [ensemble_weights[alg] for alg in train.ml_models] 
        
        train.preds = load_predictions(f'{output_path}/predictions')
        train.get_evaluation_scores(model_dir=output_path, splits=['Valid', 'Test'], ensemble_weights=ensemble_weights, 
                                    display_ci=True, save_ci=True, verbose=False)
    if subgroup_ci:
        target_keyword = f'_within_{days}days'
        prep = PrepDataEDHD()
        model_data = prep.get_data(main_dir, target_keyword)
        for algorithm in train.ml_models:
            subgroup_performance_summary(output_path, algorithm, model_data, train, display_ci=True, save_ci=True)

def main_retrain(days=30):
    main_dir = f'{root_path}/ED-H-D'
    output_path = f'{main_dir}/models/ML/within_{days}_days'
    train = get_train(main_dir, output_path, days=days)
    
    # Retrain model using best parameters
    best_params = {}
    for algorithm in train.ml_models:
        filename = f'{output_path}/best_params/{algorithm}_classifier_best_param.pkl'
        with open(filename, 'rb') as file:
            best_param = pickle.load(file)
        best_params[algorithm] = best_param

    for algorithm, model in tqdm.tqdm(train.ml_models.items()):
        if algorithm in []: continue # put the algorithms already trained in this list
        best_param = best_params[algorithm]
        if algorithm == 'NN': 
            best_param['max_iter'] = 100
            best_param['verbose'] = True
        train.train_model_with_best_param(algorithm, model, best_param, save_dir=output_path)
        print(f'{algorithm} training completed!')
        
def main_bayesopt(days=30):
    main_dir = f'{root_path}/ED-H-D'
    output_path = f'{main_dir}/models/ML/within_{days}_days'
    train = get_train(main_dir, output_path, days=days)
    
    # Conduct Baysian Optimization
    best_params = {}
    for algorithm, model in train.ml_models.items():
        if algorithm in []: continue # put the algorithms already trained and tuned in this list
        best_param = train.bayesopt(algorithm, save_dir=f'{output_path}/best_params')
        best_params[algorithm] = best_param
        if algorithm == 'NN':
            best_param['max_iter'] = 100
            best_param['verbose'] = True
        train.train_model_with_best_param(algorithm, model, best_param, save_dir=output_path)

def main_gru_bayesopt(days=30):
    main_dir = f'{root_path}/ED-H-D'
    output_path = f'{main_dir}/models/GRU/within_{days}_days'
    train = get_train(main_dir, output_path, days=days, gru=True)
    
    # Conduct Baysian Optimization
    best_param = train.bayesopt('gru', save_dir=f'{output_path}/hyperparam_tuning')
    
    # Train final model using the best parameters
    filename = f'{output_path}/hyperparam_tuning/GRU_classifier_best_param.pkl'
    with open(filename, 'rb') as file:
        best_param = pickle.load(file)

    save_path = f'{output_path}/gru_classifier'
    model, train_losses, valid_losses, train_scores, valid_scores = train.train_classification(save=True, save_path=save_path, 
                                                                                               **best_param)
    np.save(f"{output_path}/loss_and_acc/train_losses.npy", train_losses)
    np.save(f"{output_path}/loss_and_acc/valid_losses.npy", valid_losses)
    np.save(f"{output_path}/loss_and_acc/train_scores.npy", train_scores)
    np.save(f"{output_path}/loss_and_acc/valid_scores.npy", valid_scores)

if __name__ == '__main__':
    main_ci(subgroup_ci=True)
