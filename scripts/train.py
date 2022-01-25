import os
import tqdm
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization

from scripts.utilities import (twolevel, get_clean_variable_names, load_ml_model, compute_bootstrap_scores, get_bbox)
from scripts.config import (root_path, plus_minus, blood_types, cytopenia_thresholds,
                            nn_solvers, nn_activations, calib_param, calib_param_logistic)
from scripts.preprocess import (split_and_parallelize)

from sklearn.metrics import (classification_report, accuracy_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, 
                             average_precision_score, precision_recall_curve,
                             confusion_matrix)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

torch.manual_seed(0)
np.random.seed(0)

class Train:
    """
    Train machine learning models
    Employ model calibration and Bayesian Optimization
    """
    def __init__(self, dataset, clip_thresholds=None, n_jobs=32):
        self.ml_models = {"LR": LogisticRegression, # L2 Regularized Logistic Regression
                          "XGB": XGBClassifier, # Extreme Gradient Boostring
                          "RF": RandomForestClassifier,
                          "NN": MLPClassifier} # Multilayer perceptron (aka neural network)
        
        self.model_tuning_config = {'LR': (self.lr_evaluate, 
                                           {'init_points': 3, 'n_iter': 10}, 
                                           {'C': (0, 1)}),
                                    'XGB': (self.xgb_evaluate, 
                                            {'init_points': 5, 'n_iter': 25}, 
                                            {'learning_rate': (0.001, 0.1),
                                             'n_estimators': (50, 200),
                                             'max_depth': (3, 7),
                                             'gamma': (0, 1),
                                             'reg_lambda': (0, 1)}),
                                    'RF': (self.rf_evaluate, 
                                           {'init_points': 3, 'n_iter': 20}, 
                                           {'n_estimators': (50, 200),
                                            'max_depth': (3, 7),
                                            'max_features': (0.01, 1)}),
                                    'NN': (self.nn_evaluate, 
                                           {'init_points': 5, 'n_iter': 50}, 
                                           {'learning_rate_init': (0.0001, 0.1),
                                            'batch_size': (64, 512),
                                            'momentum': (0,1),
                                            'alpha': (0,1),
                                            'first_layer_size': (16, 256),
                                            'second_layer_size': (16, 256),
                                            'third_layer_size': (16, 256),
                                            'solver': (0, len(nn_solvers)),
                                            'activation': (0, len(nn_activations))}),
                                    'ENS': (self.ens_evaluate, 
                                            {'init_points': 4, 'n_iter': 30}, 
                                            {alg: (0, 1) for alg in self.ml_models})}
        
        self.X_train, self.Y_train, self.X_valid, self.Y_valid, self.X_test, self.Y_test = dataset
        self.data_splits = {'Train': (self.X_train, self.Y_train),
                            'Valid': (self.X_valid, self.Y_valid),
                            'Test': (self.X_test, self.Y_test)}
        self.preds = {split: {algorithm: None for algorithm in self.ml_models} for split in self.data_splits}
        self.clip_thresholds = clip_thresholds
        self.target_types = self.Y_train.columns.tolist()
        self.n_jobs = n_jobs
    
    def get_LR_model(self, C, max_iter=1000):
        params = {'C': C, 
                  'class_weight': 'balanced',
                  'max_iter': max_iter,
                  'random_state': 42}
        model = MultiOutputClassifier(CalibratedClassifierCV(self.ml_models['LR'](**params), **calib_param_logistic))
        return model
    
    # weight for positive examples to account for imbalanced dataset
    # scale_pos_weight = [neg_count/pos_count for index, (neg_count, pos_count) in Y_distribution['Train'].iterrows()]
    # min_child_weight = max(scale_pos_weight) * 6 # can't have less than 6 samples in a leaf node
    def get_XGB_model(self, learning_rate, n_estimators, max_depth, gamma, reg_lambda):
        params = {'learning_rate': learning_rate, 
                  'n_estimators': int(n_estimators), 
                  'max_depth': int(max_depth),
                  'gamma': gamma, 
                  'reg_lambda': reg_lambda,
                  # 'scale_pos_weight': scale_pos_weight,
                  # 'min_child_weight': min_child_weight, # set to 6 if not using scale_pos_weight
                  'min_child_weight': 6,
                  'verbosity': 0,
                  'use_label_encoder': False,
                  'random_state': 42,
                  'n_jobs': self.n_jobs,
                 }
        model = MultiOutputClassifier(CalibratedClassifierCV(self.ml_models['XGB'](**params), **calib_param))
        return model
    
    def get_RF_model(self, n_estimators, max_depth, max_features):
        params = {'n_estimators': int(n_estimators),
                  'max_depth': int(max_depth),
                  'max_features': max_features,
                  'min_samples_leaf': 6, # can't allow leaf node to have less than 6 samples
                  'class_weight': 'balanced_subsample',
                  'random_state': 42,
                  'n_jobs': self.n_jobs}
        model = MultiOutputClassifier(CalibratedClassifierCV(self.ml_models['RF'](**params), **calib_param))
        return model
    
    def get_NN_model(self, learning_rate_init, batch_size, momentum, alpha, first_layer_size, second_layer_size, third_layer_size,
                   solver, activation, max_iter=20, verbose=False):
        params = {'learning_rate_init': learning_rate_init,
                  'batch_size': int(batch_size),
                  'momentum': momentum,
                  'alpha': alpha,
                  'hidden_layer_sizes': (int(first_layer_size), int(second_layer_size), int(third_layer_size)),
                  'solver': nn_solvers[int(np.floor(solver))],
                  'activation': nn_activations[int(np.floor(activation))],
                  'max_iter': max_iter,
                  'verbose': verbose,
                  'tol': 1e-3,
                  'random_state': 42}
        # model = MultiOutputClassifier(CalibratedClassifierCV(ml_models['NN'](**params), **calib_param)) # 3 MLP, each outputs 1 value
        model = self.ml_models['NN'](**params) # 1 MLP, outputs 3 values
        return model

    def evaluate(self, model, eval_NN=False):
        model.fit(self.X_train, self.Y_train.astype(int)) # astype int because XGB throws a fit if you don't do it
        pred_prob = model.predict_proba(self.X_valid)
        result = []
        for i, target_type in enumerate(self.target_types):
            Y_true = self.Y_valid[target_type]
            Y_pred_prob = pred_prob[:, i] if eval_NN else pred_prob[i][:, 1]
            result.append(roc_auc_score(Y_true, Y_pred_prob))
        return np.mean(result)

    def lr_evaluate(self, C):
        model = self.get_LR_model(C)
        return self.evaluate(model)

    # weight for positive examples to account for imbalanced dataset
    # scale_pos_weight = [neg_count/pos_count for index, (neg_count, pos_count) in Y_distribution['Train'].iterrows()]
    # min_child_weight = max(scale_pos_weight) * 6 # can't have less than 6 samples in a leaf node
    def xgb_evaluate(self, learning_rate, n_estimators, max_depth, gamma, reg_lambda):
        model = self.get_XGB_model(learning_rate, n_estimators, max_depth, gamma, reg_lambda)
        return self.evaluate(model)

    def rf_evaluate(self, n_estimators, max_depth, max_features):
        model = self.get_RF_model(n_estimators, max_depth, max_features)
        return self.evaluate(model)

    def nn_evaluate(self, learning_rate_init, batch_size, momentum, alpha, first_layer_size, second_layer_size, third_layer_size,
                   solver, activation):
        model = self.get_NN_model(learning_rate_init, batch_size, momentum, alpha, first_layer_size, second_layer_size, third_layer_size, solver, activation)
        return self.evaluate(model, eval_NN=True)
    
    def ens_evaluate(self, **kwargs):
        """The other models must have already been trained and its predictions computed
        """
        # get model weights
        weights = [kwargs[algorithm] for algorithm in self.ml_models]
        # get ensemble predictions
        pred_prob = [self.preds['Valid'][algorithm] for algorithm in self.ml_models]
        pred_prob = np.average(pred_prob, axis=0, weights=weights)
        # get results
        result = []
        for i, target_type in enumerate(self.target_types):
            Y_true = self.Y_valid[target_type]
            Y_pred_prob = pred_prob[i]
            result.append(roc_auc_score(Y_true, Y_pred_prob))
        return np.mean(result)
    
    def convert_some_params_to_int(self, best_param):
        for param in ['max_depth', 'batch_size', 'n_estimators',
                      'first_layer_size', 'second_layer_size', 'third_layer_size']:
            if param in best_param:
                best_param[param] = int(best_param[param])
        return best_param

    def train_model_with_best_param(self, algorithm, model, best_param, save_dir):
        model = eval(f'self.get_{algorithm}_model(**best_param)')
        model.fit(self.X_train, self.Y_train)

        # Save the model
        model_filename = f'{save_dir}/{algorithm}_classifier.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)

        return model

    def bayesopt(self, algorithm, save_dir):
        # Conduct Bayesian Optimization
        evaluate_function, optim_config, hyperparam_config = self.model_tuning_config[algorithm]
        bo = BayesianOptimization(evaluate_function, hyperparam_config, random_state=42)
        bo.maximize(acq='ei', **optim_config)
        best_param = bo.max['params']
        best_param = self.convert_some_params_to_int(best_param)
        print(f'Finished finding best hyperparameters for {algorithm}')
        print('Best param:', best_param)

        # Save the best hyperparameters
        param_filename = f'{save_dir}/{algorithm}_classifier_best_param.pkl'
        with open(param_filename, 'wb') as file:    
            pickle.dump(best_param, file)

        return best_param
    
    def predict(self, model, split, algorithm, ensemble_weights=None):
        if ensemble_weights is None: 
            ensemble_weights = [1, 1, 1, 1]
            
        X, Y = self.data_splits[split]
        if algorithm not in self.preds[split] or self.preds[split][algorithm] is None:
            if algorithm == 'ENS':
                # compute ensemble predictions by soft vote
                pred = [self.preds[split][algorithm] for algorithm in self.ml_models]
                pred = np.average(pred, axis=0, weights=ensemble_weights)
            else:
                pred = model.predict_proba(X)
                # format it to be row=target, columns=chemo_sessions 
                # [:, :, 1] - first column is prob of false, second column is prob of true
                pred =  pred.T if algorithm == 'NN' else np.array(pred)[:, :, 1]
            self.preds[split][algorithm] = pred
        else:
            pred = self.preds[split][algorithm]
        return pred, Y

    def get_bootstrapped_scores(self, Y_true, Y_pred_prob, algorithm, col, model_dir, ci_df, n_bootstraps=10000):
        split, target_type = col
        ci_index = f'{algorithm}_{split}_{target_type}'
        if ci_index not in ci_df.index:
            auc_scores = compute_bootstrap_scores(Y_true.reset_index(drop=True), Y_pred_prob, n_bootstraps=n_bootstraps)
            print(f'Completed bootstrap computations for {ci_index}')
            auroc_scores, auprc_scores = np.array(auc_scores).T
            ci_df.loc[(ci_index, 'AUROC'), range(n_bootstraps)] = auroc_scores
            ci_df.loc[(ci_index, 'AUPRC'), range(n_bootstraps)] = auprc_scores
        return ci_df.loc[ci_index].values
        
    def get_confidence_interval(self, Y_true, Y_pred_prob, algorithm, col, score_df, model_dir, ci_df):
        auroc_scores, auprc_scores = self.get_bootstrapped_scores(Y_true, Y_pred_prob, algorithm, col, model_dir, ci_df)
        for name, scores in [('AUROC Score', auroc_scores), ('AUPRC Score', auprc_scores)]:
            confidence_interval = 1.96 * scores.std() / np.sqrt(len(scores))
            confidence_interval = np.round(confidence_interval * 100, 4) # convert it to percentage since number is so small
            score_df.loc[(algorithm, name), col] = f'{score_df.loc[(algorithm, name), col]} {plus_minus} {confidence_interval}%'
        return score_df
    
    def ensemble_score(self, model_dir, score_df, ci_df, splits=None, target_types=None, display_ci=False, 
                       load_ci=False, ensemble_weights=None, algorithm='ENS', digit_round=3):
        """Evaluate ensemble model using soft vote mechanism (averaging the 4 model predictions)
        This gives more weight to higher confident models
        """
        if target_types is None: target_types = self.target_types
        if splits is None: splits = self.data_splits.keys()
        
        for split in splits:
            pred_prob, Y = self.predict(None, split, algorithm, ensemble_weights=ensemble_weights)
            for target_type in target_types:
                idx = self.target_types.index(target_type)
                Y_true = Y[target_type]
                Y_pred_prob = pred_prob[idx]
                col = (split, target_type)
                score_df.loc[(algorithm, 'AUROC Score'), col] = np.round(roc_auc_score(Y_true, Y_pred_prob), digit_round)
                score_df.loc[(algorithm, 'AUPRC Score'), col] = np.round(average_precision_score(Y_true, Y_pred_prob), digit_round)
                if display_ci:
                    score_df = self.get_confidence_interval(Y_true, Y_pred_prob, algorithm, col, score_df, model_dir, ci_df)
        return score_df
    
    def cyto_baseline_score(self, score_df, splits=['Train', 'Valid', 'Test'], target_types=None):
        if self.clip_thresholds is None: 
            raise ValueError('Please intialize clip thresholds')
            
        if target_types is None: 
            target_types = self.target_types
            
        # Baseline Model - Predict Previous Value
        for metric in ['Acc', 'Precision', 'Recall', 'F1 Score']:
            score_df.loc[('Baseline - Prev', metric), :] = np.nan

        for split in splits:
            X, Y = self.data_splits[split]
            for target_type in target_types:
                blood_type = target_type.split(' < ')[0]
                bt_max = self.clip_thresholds[f'baseline_{blood_type}_count'].max()
                bt_min = self.clip_thresholds[f'baseline_{blood_type}_count'].min()
                cols = Y.columns
                col = cols[cols.str.contains(blood_type)]
                Y_true = Y[col]
                Y_pred = X[f'baseline_{blood_type}_count'] < (cytopenia_thresholds[blood_type] - bt_min) / (bt_max - bt_min)
                report = classification_report(Y_true, Y_pred, output_dict=True)

                score_df.loc[('Baseline - Prev', 'Acc'), (split, target_type)] = report['accuracy']

                # predicted true positive over all predicted positive
                score_df.loc[('Baseline - Prev', 'Precision'), (split, target_type)] = report['True']['precision']

                # predicted true positive over all true positive (aka senstivity)
                score_df.loc[('Baseline - Prev', 'Recall'), (split, target_type)] = report['True']['recall']

                # 2*precision*recall / (precision + recall)
                score_df.loc[('Baseline - Prev', 'F1 Score'), (split, target_type)] = report['True']['f1-score']

        return score_df

    def get_evaluation_scores(self, model_dir, splits=None, target_types=None, get_baseline=False, get_ensemble=True, 
                              ensemble_weights=None, display_ci=False, load_ci=False, save_ci=False, save_score=True, 
                              pred_threshold=0.5, digit_round=3,verbose=True):
        """Evaluate the best models, compute the AUROC, AUPRC, F1, etc scores.
        
        Args:
            display_ci (bool): display confidence interval for AUROC and AUPRC
            load_ci (bool): load saved bootstrapped AUROC and AUPRC scores for computing confidence interval
        """    
        if target_types is None: target_types = self.target_types
        if splits is None: splits = self.data_splits.keys()
            
        score_df = pd.DataFrame(index=twolevel, columns=twolevel)
        if get_baseline:
            score_df = self.cyto_baseline_score(score_df, splits=splits, target_types=target_types) # get score from baseline model 
            
        ci_df = pd.DataFrame(index=twolevel)
        if load_ci:
            ci_df = pd.read_csv(f'{model_dir}/confidence_interval/bootstrapped_scores.csv', index_col=[0,1])
            ci_df.columns = ci_df.columns.astype(int)

        for algorithm, model in self.ml_models.items():
            for metric in ['Acc', 'Precision', 'Recall', 'F1 Score', 'AUROC Score', 'AUPRC Score']:
                score_df.loc[(algorithm, metric), :] = np.nan

            # Load best models
            model = load_ml_model(model_dir, algorithm)
            # Load best params
            """
            filename = f'{model_dir}/{algorithm}_classifier_best_param.pkl'
            with open(filename, 'rb') as file:
                best_param = pickle.load(file)
            model = self.train_model_with_best_param(algorithm, model, best_param)
            """

            # Evaluate the model
            for split in splits:
                pred_prob, Y = self.predict(model, split, algorithm)
                for target_type in target_types:
                    idx = self.target_types.index(target_type)
                    Y_true = Y[target_type]
                    Y_pred_prob = pred_prob[idx]
                    Y_pred_bool = Y_pred_prob > pred_threshold
                    report = classification_report(Y_true, Y_pred_bool, output_dict=True, zero_division=1)
                    
                    # insert into table
                    col = (split, target_type)
                    score_df.loc[(algorithm, 'Acc'), col] = np.round(report['accuracy'], digit_round)
                    score_df.loc[(algorithm, 'Precision'), col] = np.round(report['True']['precision'], digit_round)
                    score_df.loc[(algorithm, 'Recall'), col] = np.round(report['True']['recall'], digit_round)
                    score_df.loc[(algorithm, 'F1 Score'), col] = np.round(report['True']['f1-score'], digit_round)
                    score_df.loc[(algorithm, 'AUROC Score'), col] = np.round(roc_auc_score(Y_true, Y_pred_prob), digit_round)
                    """
                    Area Under the Curve for Precision-Recall Curve
                    - appropriate for imbalanced dataset
                    - average precision is used to summarize PR Curve, although not exactly the same as AUC
                    """
                    score_df.loc[(algorithm, 'AUPRC Score'), col] = np.round(average_precision_score(Y_true, Y_pred_prob), digit_round)

                    # confusion matrix
                    if verbose and split == 'Test':
                        cm = confusion_matrix(Y_true, Y_pred_bool)
                        cm = pd.DataFrame(cm, columns=['Predicted False', 'Predicted True'], index=['Actual False', 'Actual True'])
                        print(f"\n######## {algorithm} - {split} - {target_type} #########")
                        print(cm)
                    
                    if display_ci:
                        score_df = self.get_confidence_interval(Y_true, Y_pred_prob, algorithm, col, score_df, model_dir, ci_df)
                            
        if get_ensemble:
            # gets the score for the ensemble model, which is comprised of all the models so far
            score_df = self.ensemble_score(model_dir, score_df, ci_df, splits=splits, target_types=target_types, 
                                           display_ci=display_ci, ensemble_weights=ensemble_weights, digit_round=digit_round)
        
        if save_score: score_df.to_csv(f'{model_dir}/tables/classification_results.csv')
        if save_ci: ci_df.to_csv(f'{model_dir}/confidence_interval/bootstrapped_scores.csv')
            
        return score_df

    def plot_curve(self, algorithm, model, target_types, ax, curve_type='roc', legend_location='auto', ensemble_weights=None):
        if curve_type == 'pr':
            curve_function, score_function = precision_recall_curve, average_precision_score
            label_name, xlabel, ylabel = 'AP', 'Recall', 'Precision'
        elif curve_type == 'roc':
            curve_function, score_function = roc_curve, roc_auc_score
            label_name, xlabel, ylabel = 'AUC', 'False Positive Rate', 'True Positive Rate'
            
        pred, Y = self.predict(model, 'Test', algorithm, ensemble_weights=ensemble_weights)
        for target_type in target_types:
            i = self.target_types.index(target_type)
            y_true = Y[target_type]
            y_scores = pred[i]
            x, y, thresholds = curve_function(y_true, y_scores)
            score = np.round(score_function(y_true, y_scores), 2)
            label = target_type + f' ({label_name} = {score})'
            if curve_type == 'pr': x, y = y, x
            ax.plot(x, y, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(algorithm)
        ax.legend(loc=legend_location)

    def plot_curves(self, save_dir, curve_type='roc', target_types=None, figsize=(12,9), legend_location='lower left', 
                    get_ensemble=False, ensemble_weights=None):
        nrows, ncols = 2, 2
        if get_ensemble: nrows += 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten()
        plt.subplots_adjust(hspace=0.3)
        
        if target_types is None: 
            target_types = self.target_types
            
        for idx, (algorithm, _) in enumerate(self.ml_models.items()):
            model = load_ml_model(save_dir, algorithm)
            self.plot_curve(algorithm, model, target_types, axes[idx], curve_type=curve_type, legend_location=legend_location)
        
        if get_ensemble:
            algorithm, model = 'ENS', None
            self.plot_curve(algorithm, model, target_types, axes[idx+1], curve_type=curve_type,
                            legend_location=legend_location, ensemble_weights=ensemble_weights)
            fig.delaxes(axes[-1])
            
        plt.savefig(f'{save_dir}/figures/{curve_type}_curve.jpg', bbox_inches='tight', dpi=300)

    def plot_calibs(self, save_dir, target_types=None, figsize=(12,9), savefig=True, filename='calibration_curve',
                    include_pred_hist=False, n_bins=20, calib_strategy='uniform'):
        gs = GridSpec(6, 2)
        fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(hspace=0.3)
        
        if target_types is None: 
            target_types = self.target_types

        for idx, (algorithm, model) in enumerate(self.ml_models.items()):
            model = load_ml_model(save_dir, algorithm)
            
            if include_pred_hist:
                row = int(idx / 2) * 3
                col = idx % 2
                ax = fig.add_subplot(gs[row:row+2, col])
            else:
                ax = fig.add_subplot(2, 2, idx+1)
                
            axis_max_limit = 0
            pred_prob, Y = self.predict(model, 'Valid', algorithm)
            for target_type in target_types:
                i = self.target_types.index(target_type)
                y_true = Y[target_type]
                y_pred = pred_prob[i]
                prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy=calib_strategy)
                axis_max_limit = np.max([axis_max_limit, prob_true.max(), prob_pred.max()])
                ax.plot(prob_pred, prob_true, label=target_type)
            ax.plot([0,axis_max_limit],[0,axis_max_limit],'k:', label='perfect calibration')
            ax.set_xlabel('Predicted Probability') # binned predicted probability (e.g. 0.0-0.1, 0.1-0.2, etc) - mean of each bin
            ax.set_ylabel('Empirical Probability') # fraction of positive class in each bin
            ax.set_title(algorithm)  
            ax.legend()
            
            if include_pred_hist:
                ax = fig.add_subplot(gs[row+2, col], sharex=ax)
                for target_type in target_types:
                    i = self.target_types.index(target_type)
                    y_pred = pred_prob[i]
                    ax.hist(y_pred, range=(0,axis_max_limit), bins=n_bins, label=target_type, histtype='step')
                ax.set_xlabel('Predicted Probability')
                ax.set_ylabel('Count')
                
        if include_pred_hist:
            plt.tight_layout()
            
        if savefig:
            plt.savefig(f'{save_dir}/figures/{filename}.jpg', bbox_inches='tight', dpi=300)
        
    def plot_cdf_pred(self, save_dir, target_types=None, figsize=(12,9)): 
        fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(hspace=0.3)
        
        if target_types is None: 
            target_types = self.target_types
            
        for idx, (algorithm, model) in enumerate(self.ml_models.items()):
            model = load_ml_model(save_dir, algorithm)
            pred_prob, Y = self.predict(model, 'Test', algorithm)
            N = len(Y)
            ax = fig.add_subplot(2, 2, idx+1)
            for target_type in target_types:
                i = self.target_types.index(target_type)
                x = pred_prob[i]
                x = np.sort(x)
                y = np.arange(N) / float(N)
                plt.plot(x, y, label=target_type)
            plt.title(f'CDF of {algorithm} Predictions')
            plt.xlabel('Predictions')
            plt.ylabel('Cumulative Probability')
            plt.legend()
        plt.savefig(f'{save_dir}/figures/cdf_prediction_curves.jpg', bbox_inches='tight', dpi=300)
   
    def threshold_op_points(self, model, pred_thresholds, target_types=None, algorithm='NN', digit_round=3):
        df = pd.DataFrame(columns=twolevel)
        df.index.name = 'Prediction Threshold'
        
        if target_types is None: 
            target_types = self.target_types
            
        pred_prob, Y = self.predict(model, 'Test', algorithm)
        for target_type in target_types:
            idx = self.target_types.index(target_type)
            Y_true = Y[target_type]
            Y_pred_prob = pred_prob[idx]
            for threshold in pred_thresholds:
                threshold = np.round(threshold, 2)
                Y_pred_bool = Y_pred_prob > threshold
                df.loc[threshold, (target_type, '% of Treatments with Warnings')] = Y_pred_bool.mean()
                df.loc[threshold, (target_type, 'PPV')] = precision_score(Y_true, Y_pred_bool, zero_division=1)
                df.loc[threshold, (target_type, 'Sensitivity')] = recall_score(Y_true, Y_pred_bool, zero_division=1)
                df.loc[threshold, (target_type, 'NPV')] = precision_score(~Y_true, ~Y_pred_bool, zero_division=1)
                df.loc[threshold, (target_type, 'Specificity')] = recall_score(~Y_true, ~Y_pred_bool, zero_division=1)
        df = df.round(digit_round)
        return df
    
    def all_plots_for_single_target(self, save_dir, algorithm='XGB', target_type='ACU', split='Test',
                                    n_bins=10, n_samples_for_ci=10, calib_strategy='quantile', figsize=(12,9)):
        # setup
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        axes = axes.flatten()
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        idx = self.target_types.index(target_type)
        model = None if algorithm == 'ENS' else load_ml_model(save_dir, algorithm)
        pred_prob, Y = self.predict(model, split, algorithm)
        y_true = Y[target_type]
        y_pred = pred_prob[idx]
        
        # get 95% confidence interval
        col = (split, target_type)
        ci_df = pd.read_csv(f'{save_dir}/confidence_interval/bootstrapped_scores.csv', index_col=[0,1])
        auroc_scores, auprc_scores = self.get_bootstrapped_scores(y_true, y_pred, algorithm, col, save_dir, ci_df)
        ap_lower, ap_upper = np.percentile(auprc_scores, [2.5, 97.5]).round(3)
        roc_lower, roc_upper = np.percentile(auroc_scores, [2.5, 97.5]).round(3)

        # get results
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        ap_score = np.round(average_precision_score(y_true, y_pred), 3)
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_score = np.round(roc_auc_score(y_true, y_pred), 3)
        
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins*n_samples_for_ci, strategy='quantile')
        prob_true, prob_pred = prob_true.reshape(n_bins, n_samples_for_ci), prob_pred.reshape(n_bins, n_samples_for_ci)
        mean_prob_true, mean_prob_pred = prob_true.mean(axis=1), prob_pred.mean(axis=1)
        lower_limit, upper_limit = prob_true[:, 0] - mean_prob_true, prob_true[:, -1] - mean_prob_true
        yerr = np.array([abs(lower_limit), upper_limit])
        axis_max_limit = max(mean_prob_true.max(), mean_prob_pred.max())
        
        N = len(Y)
        x, y = np.sort(y_pred), np.arange(N) / float(N)

        # plot
        axes[0].plot(recall, precision, label=f'AUPRC={ap_score} (95% CI: {ap_lower}-{ap_upper})')
        axes[0].set_ylim(0, 1)
        axes[1].plot(fpr, tpr, label=f'AUROC={roc_score} (95% CI: {roc_lower}-{roc_upper})')
        axes[2].errorbar(mean_prob_pred, mean_prob_true, yerr=yerr, capsize=5.0, ecolor='firebrick')
        axes[2].plot([0,axis_max_limit],[0,axis_max_limit],'k:', label='Perfect Calibration')
        axes[3].plot(x, y)
        
        # label
        labels = [('Sensitivity', 'Positive Predictive Value', 'pr_curves', True),
                  ('1 - Specificity', 'Sensitivity', 'roc_curves', True),
                  ('Predicted Probability', 'Empirical Probability', 'calibration_curves', False),
                  (f'Predicted Probability of {target_type}', 'Cumulative Proportion of Predictions', 'cdf_prediction_curves', False)]
        for idx, (xlabel, ylabel, filename, remove_legend_line) in enumerate(labels):
            axes[idx].set_xlabel(xlabel)
            axes[idx].set_ylabel(ylabel)
            leg = axes[idx].legend(loc='lower right', frameon=False)
            if remove_legend_line: leg.legendHandles[0].set_linewidth(0)
            fig.savefig(f'{save_dir}/figures/{filename}_{algorithm}_{target_type}.jpg', bbox_inches=get_bbox(axes[idx], fig), dpi=300) 
        plt.savefig(f'{save_dir}/figures/all_plots_{algorithm}_{target_type}.jpg', bbox_inches='tight', dpi=300)
        
    def get_LR_weights(self, LR_model, target_types=None):
        if target_types is None:
            target_types = self.target_types
        cols = get_clean_variable_names(self.X_train.columns)
        LR_weights_df = pd.DataFrame(index=cols, columns=target_types)
        for idx, target_type in enumerate(target_types):
            estimator = LR_model.estimators_[idx]
            coefs = []
            for i in range(3):
                coef = estimator.calibrated_classifiers_[i].base_estimator.coef_
                coefs.append(coef)
            mean_coef = np.array(coefs).mean(axis=0)[0]
            LR_weights_df[target_type] = mean_coef
        LR_weights_df = LR_weights_df.round(3)
        return LR_weights_df

# Gated Recurrent Unit Model
class GRU(nn.Module):
    def __init__(self, n_features, n_targets, hidden_size, hidden_layers, batch_size, dropout, pad_value):
        super().__init__()
        self.n_features = n_features
        self.n_targets = n_targets
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.pad_value = pad_value
        self.dropout = dropout
        self.rnn_layers = nn.GRU(input_size=self.n_features, hidden_size=self.hidden_size, 
                                 num_layers=self.hidden_layers, dropout=self.dropout, batch_first=True)
        self.linear = nn.Linear(in_features=self.hidden_size, 
                                out_features=self.n_targets)

    def forward(self, packed_inputs):
        out_packed, _ = self.rnn_layers(packed_inputs)
        output, lengths = pad_packed_sequence(out_packed, batch_first=True, padding_value=self.pad_value)
        output = self.linear(output)
        return output
    
    def init_hidden(self):
        return torch.zeroes(self.hidden_layers, self.batch_size, self.hidden_size)
    
class SeqData(TensorDataset):
    def __init__(self, mapping, ids):
        self.mapping = mapping
        self.ids = ids
                
    def __getitem__(self, index):
        sample = self.ids[index]
        X, Y = self.mapping[sample]
        features_tensor = torch.Tensor(X.values)
        target_tensor = torch.Tensor(Y.values)
        return features_tensor, target_tensor
    
    def __len__(self):
        return(len(self.ids))
    
class TrainGRU(Train):
    def __init__(self, dataset, output_path=None, clip_thresholds=None, n_jobs=32):
        super(TrainGRU, self).__init__(dataset, clip_thresholds, n_jobs)
        self.model_tuning_config = {'gru': (self.gru_evaluate, 
                                           {'init_points': 3, 'n_iter': 50}, 
                                           {'batch_size': (16, 512),
                                            'learning_rate': (0.0001, 0.01),
                                            'hidden_size': (10, 200),
                                            'hidden_layers': (1, 5),
                                            'dropout': (0.0, 0.9)})}
        self.n_features = self.X_train.shape[1] - 1 # -1 for ikn
        self.n_targets = self.Y_train.shape[1]
        self.train_dataset = self.transform_to_tensor_dataset(self.X_train, self.Y_train)
        self.valid_dataset = self.transform_to_tensor_dataset(self.X_valid, self.Y_valid)
        self.test_dataset = self.transform_to_tensor_dataset(self.X_test, self.Y_test)
        self.data_splits = {'Train': self.train_dataset, 'Valid': self.valid_dataset, 'Test': self.test_dataset}
        self.output_path = output_path
        
    def transform_to_tensor_dataset(self, X, Y):
        X = X.astype(float)
        mapping = {}
        for ikn, group in tqdm.tqdm(X.groupby('ikn')):
            group = group.drop(columns=['ikn'])
            mapping[ikn] = (group, Y.loc[group.index])
            
        return SeqData(mapping=mapping, ids=X['ikn'].unique())
    
    def evaluate(self, model, loader, criterion, pad_value):
        total_loss = 0
        total_score = 0
        for i, batch in enumerate(loader):
            inputs, targets = tuple(zip(*batch))
            seq_lengths = list(map(len, inputs))
            padded = pad_sequence(inputs, batch_first=True, padding_value=pad_value)
            padded_packed = pack_padded_sequence(padded, seq_lengths, batch_first=True, enforce_sorted=False).float()
            padded_targets = pad_sequence(targets, batch_first=True, padding_value=pad_value)
            targets = torch.cat(targets).float()
            if torch.cuda.is_available():
                padded_packed = padded_packed.cuda()
                targets = targets.cuda()
            preds = model(padded_packed)
            flag = (padded_targets != pad_value)
            preds_res = preds[flag].reshape(-1, self.n_targets).split(seq_lengths)
            preds = torch.cat(preds_res).float()
            loss = criterion(preds, targets)
            loss = loss.mean(axis=0)
            total_loss += loss
            preds = torch.sigmoid(preds)
            preds = preds > 0.5
            if torch.cuda.is_available():
                preds = preds.cpu().detach().numpy()
                targets = targets.cpu().detach().numpy()
            total_score += np.array([accuracy_score(targets[:, i], preds[:, i]) for i in range(self.n_targets)])
        return total_loss.cpu().detach().numpy() / (i+1), total_score/(i+1)
    
    def train_classification(self, batch_size=512, pad_value=-999, epochs=200, learning_rate=0.001, 
                hidden_size=20, hidden_layers=3, dropout=0.5, decay=0, save=False, save_path=None, early_stopping=20):
        model = GRU(n_features=self.n_features, n_targets=self.n_targets, hidden_size=hidden_size, hidden_layers=hidden_layers,
                    batch_size=batch_size, dropout=dropout, pad_value=pad_value)
        if torch.cuda.is_available():
            model.cuda()

        train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x:x)
        valid_loader = DataLoader(dataset=self.valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x:x)

        best_val_loss = np.inf
        best_model_param = None
        torch.manual_seed(42)

        criterion = nn.BCEWithLogitsLoss(reduction='none')
        """
        This loss criterion COMBINES Sigmoid and BCELoss. 
        This is more numerically stable than using Sigmoid followed by BCELoss.
        AS a result, the model does not use a Simgoid layer at the end. 
        The model prediction output will not be bounded from (0, 1).
        In order to bound the model prediction, you must Sigmoid the model output.
        """
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)

        train_losses = np.zeros((epochs, self.n_targets))
        valid_losses = np.zeros((epochs, self.n_targets))
        train_scores = np.zeros((epochs, self.n_targets)) # acc score
        valid_scores = np.zeros((epochs, self.n_targets)) # acc score
        counter = 0 # for early stopping

        for epoch in range(epochs):
            train_loss = 0
            train_score = 0
            for i, batch in enumerate(train_loader):
                inputs, targets = tuple(zip(*batch)) # each is a tuple of tensors
                seq_lengths = list(map(len, inputs)) # get length of each sequence

                # Reformat to allow for variable sequence lengths with torch helpers
                padded = pad_sequence(inputs, batch_first=True, padding_value=pad_value)
                padded_packed = pack_padded_sequence(padded, seq_lengths, batch_first=True, enforce_sorted=False).float()
                padded_targets = pad_sequence(targets, batch_first=True, padding_value=pad_value)
                targets = torch.cat(targets).float()

                if torch.cuda.is_available():
                    padded_packed = padded_packed.cuda()
                    targets = targets.cuda()

                # Make predictions
                preds = model(padded_packed)

                # Unpad predictions based on target lengths
                flag = (padded_targets != pad_value)
                preds_res = preds[flag].reshape(-1, self.n_targets).split(seq_lengths)
                preds = torch.cat(preds_res).float()

                # Calculate loss
                loss = criterion(preds, targets)

                loss = loss.mean(axis=0)
                train_loss += loss

                # Bound the model prediction
                preds = torch.sigmoid(preds)

                preds = preds > 0.5
                if torch.cuda.is_available():
                    preds = preds.cpu().detach().numpy()
                    targets = targets.cpu().detach().numpy()
                train_score += np.array([accuracy_score(targets[:, i], preds[:, i]) for i in range(self.n_targets)])

                loss = loss.mean()
                loss.backward() # back propagation, compute gradients
                optimizer.step() # apply gradients
                optimizer.zero_grad() # clear gradients for next train

            train_losses[epoch] = train_loss.cpu().detach().numpy()/(i+1)
            train_scores[epoch] = train_score/(i+1)
            valid_losses[epoch], valid_scores[epoch] = self.evaluate(model, valid_loader, criterion, pad_value)
            statement = f"Epoch: {epoch+1}, \
Train Loss: {np.round(train_losses[epoch].mean(),4)}, \
Valid Loss: {np.round(valid_losses[epoch].mean(),4)}, \
Train Accuracy: {np.round(train_scores[epoch].mean(), 4)}, \
Valid Accuracy: {np.round(valid_scores[epoch].mean(),4)}"
            print(statement)

            if valid_losses[epoch].mean() < best_val_loss:
                print('Saving Best Model')
                best_val_loss = valid_losses[epoch].mean()
                best_model_param = model.state_dict()
                counter = 0

            # early stopping
            if counter > early_stopping: 
                train_losses = train_losses[:epoch+1]
                valid_losses = valid_losses[:epoch+1]
                train_scores = train_scores[:epoch+1]
                valid_scores = valid_scores[:epoch+1]
                break
            counter += 1

        if save:
            print(f'Writing best model parameter to {save_path}')
            torch.save(best_model_param, save_path)

        return model, train_losses, valid_losses, train_scores, valid_scores
    
    def plot_training_curve(self, train_losses, valid_losses, train_scores, valid_scores, 
                            figsize=(12,9), save=False, save_path=None):
        fig = plt.figure(figsize=figsize)
        data = {'Training Loss': train_losses, 'Validation Loss': valid_losses, 
                'Training Accuracy': train_scores, 'Validation Accuracy': valid_scores}
        plt.subplots_adjust(hspace=0.3)
        for i, (title, value) in enumerate(data.items()):
            ax = fig.add_subplot(2, 2, i+1)
            plt.plot(range(len(value)), value)
            plt.title(title)
            plt.legend(self.target_types)
        if save: plt.savefig(save_path)
        plt.show()
        
    def get_model_predictions(self, model, dataset, pad_value=-999):
        loader = DataLoader(dataset=dataset, batch_size=10000, shuffle=False, collate_fn=lambda x:x)
        pred_arr = np.empty([0, self.n_targets])
        target_arr = np.empty([0, self.n_targets])
        for i, batch in enumerate(loader):
            inputs, targets = tuple(zip(*batch))
            seq_lengths = list(map(len, inputs))
            padded = pad_sequence(inputs, batch_first=True, padding_value=pad_value)
            padded_packed = pack_padded_sequence(padded, seq_lengths, batch_first=True, enforce_sorted=False).float()
            padded_targets = pad_sequence(targets, batch_first=True, padding_value=pad_value)
            targets = torch.cat(targets).float()
            if torch.cuda.is_available():
                padded_packed = padded_packed.cuda()
                targets = targets.cuda()
            with torch.no_grad(): # we are only doing feedforward, no need to store prev computations/gradients, reduce memory usage
                preds = model(padded_packed)
            flag = (padded_targets != pad_value)
            preds_res = preds[flag].reshape(-1, self.n_targets).split(seq_lengths)
            preds = torch.cat(preds_res).float()
            preds = torch.sigmoid(preds)

            if torch.cuda.is_available():
                preds = preds.cpu().detach().numpy()
                targets = targets.cpu().detach().numpy()

            pred_arr = np.concatenate([pred_arr, preds])
            target_arr = np.concatenate([target_arr, targets])
        
        # garbage collection
        del preds, targets, padded_packed
        torch.cuda.empty_cache()
        
        return pred_arr, target_arr

    def get_model_scores(self, model, target_types=None, save=False, save_path=None, 
                         splits=['Train', 'Valid', 'Test'], verbose=True, algorithm='GRU'):
        score_df = pd.DataFrame(index=twolevel, columns=twolevel)
            
        if target_types is None: 
            target_types = self.target_types

        for split in splits:
            dataset = self.data_splits[split]
            pred, target = self.get_model_predictions(model, dataset)
            for target_type in target_types:
                idx = self.target_types.index(target_type)
                Y_true = target[:, idx]
                Y_pred_prob = pred[:, idx]
                Y_pred_bool = Y_pred_prob > 0.5
                report = classification_report(Y_true, Y_pred_bool, output_dict=True, zero_division=1)

                score_df.loc[(algorithm, 'Acc'), (split, target_type)] = report['accuracy']
                score_df.loc[(algorithm, 'Precision'), (split, target_type)] = report['1.0']['precision']
                score_df.loc[(algorithm, 'Recall'), (split, target_type)] = report['1.0']['recall']
                score_df.loc[(algorithm, 'F1 Score'), (split, target_type)] = report['1.0']['f1-score']
                score_df.loc[(algorithm, 'AUROC Score'), (split, target_type)] = roc_auc_score(Y_true, Y_pred_prob)
                score_df.loc[(algorithm, 'AUPRC Score'), (split, target_type)] = average_precision_score(Y_true, Y_pred_prob)

                # confusion matrix
                if verbose and split == 'Test':
                    cm = confusion_matrix(Y_true, Y_pred_bool)
                    cm = pd.DataFrame(cm, columns=['Predicted False', 'Predicted True'], index=['Actual False', 'Actual True'])
                    print(f"\n############ {split} - {target_type} #############")
                    print(cm)

        score_df = pd.DataFrame(score_df.values.astype(float).round(4), index=score_df.index, columns=score_df.columns)
        if save:
            score_df.to_csv(save_path, index=False)

        return score_df
    
    def get_evaluation_score(self, model, dataset, pad_value=-999):
        # compute total mean auc score for all target types on validation set
        pred, target = self.get_model_predictions(model, dataset)
        auc_score = [roc_auc_score(target[:, i], pred[:, i]) for i in range(self.n_targets)]
        return np.mean(auc_score)

    def gru_evaluate(self, batch_size, learning_rate, hidden_size, hidden_layers, dropout):
        bs = int(batch_size)
        lr = learning_rate
        hs = int(hidden_size)
        hl = int(hidden_layers)
        dp = dropout
        params = {'batch_size': bs,
                  'learning_rate': lr,
                  'hidden_size': hs,
                  'hidden_layers': hl,
                  'dropout': dp,
                  'epochs': 10,
                  'save': True,
                  'save_path': f'{self.output_path}/hyperparam_tuning/gru_classifier_bs{bs}lr{lr}hs{hs}hl{hl}dp{dp}'}
        model, _, _, _, _ = self.train_classification(**params)
        return self.get_evaluation_score(model, self.valid_dataset)
    
    def convert_some_params_to_int(self, best_param):
        for param in ['batch_size', 'hidden_size', 'hidden_layers']:
            if param in best_param:
                best_param[param] = int(best_param[param])
        return best_param
    
    def threshold_op_points(self, model, pred_thresholds, target_types=None, split='Test', digit_round=3):
        df = pd.DataFrame(columns=twolevel)
        df.index.name = 'Prediction Threshold'
        
        if target_types is None: 
            target_types = self.target_types
        
        dataset = self.data_splits[split]
        pred, target = self.get_model_predictions(model, dataset)
        for target_type in target_types:
            idx = self.target_types.index(target_type)
            Y_true = target[:, idx]
            Y_pred_prob = pred[:, idx]
            for threshold in pred_thresholds:
                threshold = np.round(threshold, 2)
                Y_pred_bool = Y_pred_prob > threshold
                df.loc[threshold, (target_type, '% of Treatments with Warnings')] = Y_pred_bool.mean()
                df.loc[threshold, (target_type, 'PPV')] = precision_score(Y_true, Y_pred_bool, zero_division=1)
                df.loc[threshold, (target_type, 'Sensitivity')] = recall_score(Y_true, Y_pred_bool, zero_division=1)
                df.loc[threshold, (target_type, 'NPV')] = precision_score(~Y_true.astype(bool), ~Y_pred_bool, zero_division=1)
                df.loc[threshold, (target_type, 'Specificity')] = recall_score(~Y_true.astype(bool), ~Y_pred_bool, zero_division=1)
        df = df.round(digit_round)
        return df