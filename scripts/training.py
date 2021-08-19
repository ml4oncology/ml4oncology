"""
========================================================================
© 2021 Institute for Clinical Evaluative Sciences. All rights reserved.

TERMS OF USE:
##Not for distribution.## This code and data is provided to the user solely for its own non-commercial use by individuals and/or not-for-profit corporations. User shall not distribute without express written permission from the Institute for Clinical Evaluative Sciences.

##Not-for-profit.## This code and data may not be used in connection with profit generating activities.

##No liability.## The Institute for Clinical Evaluative Sciences makes no warranty or representation regarding the fitness, quality or reliability of this code and data.

##No Support.## The Institute for Clinical Evaluative Sciences will not provide any technological, educational or informational support in connection with the use of this code and data.

##Warning.## By receiving this code and data, user accepts these terms, and uses the code and data, solely at its own risk.
========================================================================
"""
""" Train machine learning models on population-based administrative dataset
"""
import sys
for i, p in enumerate(sys.path):
    sys.path[i] = sys.path[i].replace("/software/anaconda/3/", "/MY/PATH/.conda/envs/myenv/")
import os
import tqdm
import pandas as pd
import numpy as np
import utilities as util
import warnings
import pickle
# warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (mean_squared_error,
                             classification_report, accuracy_score,
                            plot_confusion_matrix, confusion_matrix, 
                            plot_roc_curve, roc_auc_score, roc_curve, 
                            average_precision_score, plot_precision_recall_curve, precision_recall_curve)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBRegressor, XGBClassifier

from bayes_opt import BayesianOptimization

class PrepData:
    """Prepare the data for model training
    """
    def __init__(self):
        # regimens
        self.regimen_metadata = util.get_included_regimen(util.read_partially_reviewed_csv())
        self.cycle_lengths = self.regimen_metadata['cycle_length'].to_dict()

        # extra features
        extra_features = pd.read_csv('data/chemo_processed2.csv')
        self.esas_ecog_features = extra_features[['ecog_grade', 'Wellbeing','Tiredness', 'Pain', 'Shortness of Breath', 'Drowsiness', 
                                             'Lack of Appetite', 'Depression', 'Anxiety', 'Nausea']]
        cols = extra_features.columns
        extra_blood_work_cols = cols[cols.str.contains('prev')]
        self.extra_blood_work_cols = extra_blood_work_cols.drop('prev_visit')
        self.extra_blood_work_features = extra_features[self.extra_blood_work_cols]

        # blood types
        self.blood_types = ['neutrophil', 'hemoglobin', 'platelet']

    def read_data(self, blood_type):
        df = pd.read_csv(f'data/{blood_type}.csv', dtype={'curr_morph_cd': str, 'curr_topog_cd': str})
        
        # include the new blood work and questionnaire data features
        df = pd.concat([df, self.esas_ecog_features], axis=1)
        df = pd.concat([df, self.extra_blood_work_features], axis=1)

        # turn string of numbers columns into integer column 
        df = df.rename({str(i): i for i in range(-5, 29)}, axis='columns')
        
        keep_indices = []
        for regimen, group in df.groupby('regimen'):
            # set blood count measures after a regiment's respective cycle length to null
            cycle_length = int(self.cycle_lengths[regimen])
            df.loc[group.index, range(cycle_length+1, 29)] = np.nan

            # remove rows that has no blood count measure near the cycle length (within 3 day before administration day)
            if cycle_length == 28:
                cycle_length_window = range(cycle_length-2, cycle_length+1)
            else:  
                cycle_length_window = range(cycle_length-1,cycle_length+2)
            mask = (~group[cycle_length_window].isnull()).sum(axis=1) >= 1
            keep_indices += group[mask].index.tolist()
        df = df.loc[keep_indices]

        # only keep rows that have at least 2 blood count measures
        df = df[(~df[range(-5,29)].isnull()).sum(axis=1) >= 2]
        
        return df

    def get_data(self):
        data = {blood_type: read_data(blood_type) for blood_type in self.blood_types}
        
        # keep only rows where all blood types are present
        n_indices = data['neutrophil'].index
        h_indices = data['hemoglobin'].index
        p_indices = data['platelet'].index
        keep_indices = n_indices[n_indices.isin(h_indices) & n_indices.isin(p_indices)]
        data = {blood_type: data[blood_type].loc[keep_indices] for blood_type in self.blood_types}
        return data

    def organize_data(self, data):
        """Organize data for model input
        
        NBC - neutrophil blood count
        HBC - hemoglobin blood count
        PBC - platelet blood count

        input:                               -->        MODEL         -->            output:
        regimen                                                                      last observed NBC value
        prev observed NBC value                                                      last observed HBC value
        prev observed HBC value                                                      last observed PBC value
        prev observed PBC value
        days since prev observed NBC/HBC/PBC value
        chemo cycle
        immediate new regimen
        intent of systemic treatment
        line of therapy
        lhin cd
        curr morth cd
        curr topog cd
        age 
        sex
        body surface area
        esas/ecog features
        prev observed extra blood work
        """
        drop_columns = ['visit_date', 'prev_visit', 'chemo_interval'] + list(range(-5,29))
        model_data = data['neutrophil'].drop(columns=drop_columns) # all blood types have the same values
        model_data = model_data.reset_index(drop=True) # make concatting new columns easier
        
        get_num_days_between_mes = lambda row: np.diff(np.where(~np.isnan(row))[0][-2:])

        for blood_type, df in data.items():
            values = df[range(-5,29)].values
            # get number of days between the last and the previous blood count measurement (for a single row)
            days_in_between = np.array([get_num_days_between_mes(row) for row in values])
            # get all non nan values from each row
            values = [row[~np.isnan(row)] for row in values]
            # keep only the last two values (the prev blood count mes and last blood count mes)
            values = np.array([row[-2:] for row in values])
            # combine the results
            values = np.concatenate((values, days_in_between), axis=1)
            tmp_df = pd.DataFrame(values, columns=[f'prev_{blood_type}_value', f'last_{blood_type}_value', 
                                                   f'num_days_btwn_prev_{blood_type}_value'])
            model_data = pd.concat([model_data, tmp_df], axis=1)
            
        return model_data

    def remove_outliers(self, data):
        """Remove the upper and lower 1 percentiles for the columns indicated below
        """
        cols = ['prev_neutrophil_value', 'last_neutrophil_value', 
                'prev_hemoglobin_value', 'last_hemoglobin_value', 
                'prev_platelet_value', 'last_platelet_value']
        num_rows_removed = 0
        for col in cols:
            size = len(data)
            percentile1 = data[col].quantile(0.01)
            percentile99 = data[col].quantile(0.99)
            data = data[(data[col] > percentile1) & (data[col] < percentile99)]
            # print(f'Removed outliers from column {col}, {size-len(data)} rows removed')
            num_rows_removed += size-len(data)
        print('Total number of outlier rows removed =', num_rows_removed)
        return data  

    def dummify_data(self, data):
        """Make categorical columns into one-hot encoding
        """
        data['line_of_therapy'] = data['line_of_therapy'].astype('category')
        return pd.get_dummies(data)

    def replace_missing_body_surface_area(self, data, means_train=None):
        """Replace missing body surface area with the mean based on sex
        """
        bsa = 'body_surface_area'
        means = {'female': data.loc[data['sex_F'] == 1, bsa].mean(),
                 'male': data.loc[data['sex_M'] == 1, bsa].mean()} if means_train is None else means_train
        data.loc[data['sex_F'] == 1, bsa] = data.loc[data['sex_F'] == 1, bsa].fillna(means['female'])
        data.loc[data['sex_M'] == 1, bsa] = data.loc[data['sex_M'] == 1, bsa].fillna(means['male'])
        if means_train is None:
            return data, means
        else:
            return data
        
    def replace_missing_extra_blood_work(self, data, means_train=None):
        """Mean impute missing blood work data
        """
        means = data[self.extra_blood_work_cols].mean() if means_train is None else means_train
        for col in self.extra_blood_work_cols:
            data[col] = data[col].fillna(means[col])
        if means_train is None:
            return data, means
        else:
            return data
        
    def split_data(self, data):
        """
        Split data into training, validation and test sets based on patient ids
        """
        # convert dtype object to float
        data = data.astype(float)
        
        # create training set
        gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
        train_idxs, test_idxs = next(gss.split(data, groups=data['ikn']))
        train_data = data.iloc[train_idxs]
        test_data = data.iloc[test_idxs]

        # crate validation and testing set
        gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        valid_idxs, test_idxs = next(gss.split(test_data, groups=test_data['ikn']))

        valid_data = test_data.iloc[valid_idxs]
        test_data = test_data.iloc[test_idxs]

        # sanity check - make sure there are no overlap of patients in the splits
        assert(sum(valid_data['ikn'].isin(set(train_data['ikn']))) + sum(valid_data['ikn'].isin(set(test_data['ikn']))) + 
               sum(train_data['ikn'].isin(set(valid_data['ikn']))) + sum(train_data['ikn'].isin(set(test_data['ikn']))) + 
               sum(test_data['ikn'].isin(set(train_data['ikn']))) + sum(test_data['ikn'].isin(set(valid_data['ikn']))) 
               == 0)
        print(f'Size of splits: Train:{len(train_data)}, Val:{len(valid_data)}, Test:{len(test_data)}')
        print(f"Number of patients: Train:{len(set(train_data['ikn']))}, Val:{len(set(valid_data['ikn']))}, Test:{len(set(test_data['ikn']))}")
        
        # replace missing body surface area with the mean
        train_data, means_train = self.replace_missing_body_surface_area(train_data.copy())
        print(f"Body Surface Area Mean - Female:{means_train['female']}, Male:{means_train['male']}")
        valid_data = self.replace_missing_body_surface_area(valid_data.copy(), means_train)
        test_data = self.replace_missing_body_surface_area(test_data.copy(), means_train)
        
        # mean impute the blood work data
        train_data, means_train = self.replace_missing_blood_work(train_data.copy())
        valid_data = self.replace_missing_blood_work(valid_data.copy(), means_train)
        test_data = self.replace_missing_blood_work(test_data.copy(), means_train)
        
        # normalize the splits based on training data
        train_data, minmax_train = self.normalize_data(train_data.copy())
        valid_data, minmax_valid = self.normalize_data(valid_data.copy(), minmax_train)
        test_data, minmax_test = self.normalize_data(test_data.copy(), minmax_train)
        
        # split into input features and target labels
        cols = data.columns
        feature_cols = cols[~cols.str.contains('last')]
        feature_cols = feature_cols.drop('ikn')
        target_cols = cols[cols.str.contains('last')]
        X_train, X_valid, X_test = train_data[feature_cols], valid_data[feature_cols], test_data[feature_cols]
        Y_train, Y_valid, Y_test = train_data[target_cols], valid_data[target_cols], test_data[target_cols]
        
        return [(X_train, Y_train, minmax_train), (X_valid, Y_valid, minmax_valid), (X_test, Y_test, minmax_test)]

    def standardize_data(self, data):
        scalers = {}
        for blood_type in self.blood_types:
            cols = [f'prev_{blood_type}_value', f'last_{blood_type}_value']
            scaler = StandardScaler().fit(data[cols])
            scalers[blood_type] = scaler # save for future use
            data[cols] = scaler.transform(data[cols])
        return data, scalers

    def destandardize_data(self, scalers, data):
        for blood_type in self.blood_types:
            cols = [f'prev_{blood_type}_value', f'last_{blood_type}_value']
            scaler = scalers[blood_type]
            data[cols] = scaler.inverse_transform(data[cols])
        return data

    def normalize_data(self, data, minmax_train=None):
        minmax = pd.DataFrame(index=['min', 'max'])
        
        cols = ['chemo_cycle', 'age', 'body_surface_area'] + \
                self.esas_ecog_features.columns.tolist() + \
                self.extra_blood_work_cols.tolist() + \
                [f'num_days_btwn_prev_{blood_type}_value' for blood_type in self.blood_types]
        for col in cols:
            tmp = data[col]
            minmax[col] = [tmp.min(), tmp.max()]
            maximum = minmax.loc['max', col] if minmax_train is None else minmax_train.loc['max', col]
            minimum = minmax.loc['min', col] if minmax_train is None else minmax_train.loc['min', col]
            data[col] = (tmp-minimum) / (maximum-minimum)
        
        for blood_type in self.blood_types:
            cols = [f'prev_{blood_type}_value', f'last_{blood_type}_value']
            tmp = data[cols]
            minmax[blood_type] = [tmp.min().min(), tmp.max().max()]
            maximum = minmax.loc['max', blood_type] if minmax_train is None else minmax_train.loc['max', blood_type]
            minimum = minmax.loc['min', blood_type] if minmax_train is None else minmax_train.loc['min', blood_type]
            data[cols] = (tmp-minimum) / (maximum-minimum)
        
        return data, minmax

    def denormalize_data(self, data, denorm_key):
        cols = ['chemo_cycle', 'age', 'body_surface_area'] + \
                self.esas_ecog_features.columns.tolist() + \
                self.extra_blood_work_cols.tolist() + \
                [f'num_days_btwn_prev_{blood_type}_value' for blood_type in self.blood_types]
        for col in cols:
            tmp = data[col]
            scale = denorm_key[col]
            data[col] = tmp*(scale.max() - scale.min()) + scale.min()
        
        for blood_type in self.blood_types:
            cols = [f'prev_{blood_type}_value', f'last_{blood_type}_value']
            tmp = data[cols]
            scale = denorm_key[blood_type]
            data[cols] = tmp*(scale.max() - scale.min()) + scale.min()
        
        return data

    def regression_to_classification(target):
        """Convert regression labels (last blood count value) 
        to classification labels (if last blood count value is below dangerous threshold)
        """
        target[f'neutrophil < 1.5'] = target['last_neutrophil_value'] < self.neutrophil_threshold
        target[f'hemoglobin < 100'] = target['last_hemoglobin_value'] < self.hemoglobin_threshold
        target[f'platelet < 75'] = target['last_platelet_value'] < self.platelet_threshold
        target = target.drop(columns=[f'last_{blood_type}_value' for blood_type in self.blood_types])
        return target

    def prep_data(self):
        """Prep data for model input
        """
        data = self.get_data()
        model_data = self.organize_data(data)
        print(f'Size of model_data: {model_data.shape}\nNumber of unique patients: {len(set(model_data.ikn))}')
        # model_data = self.remove_outliers(model_data)
        # print(f'Size of model_data: {model_data.shape}\nNumber of unique patients: {len(set(model_data.ikn))}')
        model_data = self.dummify_data(model_data)
        print(f'Size of model_data: {model_data.shape}\nNumber of unique patients: {len(set(model_data.ikn))}')
        train, valid, test = self.split_data(model_data)
        
        # Extract the splits
        X_train, Y_train, minmax_train = train
        X_valid, Y_valid, minmax_valid = valid
        X_test, Y_test, minmax_test = test

        # Calculate normalized blood count thresholds for classification
        n_min, n_max = minmax['neutrophil']
        h_min, h_max = minmax['hemoglobin']
        p_min, p_max = minmax['platelet']
        self.neutrophil_threshold = (1.5 - n_min)/(n_max - n_min)
        self.hemoglobin_threshold = (100 - h_min)/(h_max - h_min)
        self.platelet_threshold = (75 - p_min)/(p_max - p_min)

        # Covert label to binary
        Y_train = regression_to_classification(Y_train)
        Y_valid = regression_to_classification(Y_valid)
        Y_test = regression_to_classification(Y_test)

        # Upsample the train and valid set by increasing examples where platelet < 75 by 5 folds
        indices = Y_train.index.tolist() + Y_train[Y_train['platelet < 75']].index.tolist()*5
        Y_train = Y_train.loc[indices]
        X_train = X_train.loc[indices]
        indices = Y_valid.index.tolist() + Y_valid[Y_valid['platelet < 75']].index.tolist()*5
        Y_valid = Y_valid.loc[indices]
        X_valid = X_valid.loc[indices]

        return train, valid, test

class Train(PrepData):
    """Train machine learning models
    Employ model calibration and Baysian hyperparameter optimization
    """
    def __init__(self):
        cols = pd.MultiIndex.from_product([['Train', 'Valid'], self.blood_types])
        indices = pd.MultiIndex.from_product([[], []])
        self.score_df = pd.DataFrame(index=indices, columns=cols)

        # get the data splits
        train, valid, test = self.prep_data()
        self.X_train, self.Y_train, self.minmax_train = train
        self.X_valid, self.Y_valid, self.minmax_valid = valid
        self.X_test, self.Y_test, self.minmax_test = test

        # ml models
        self.ml_models = {"LR": LogisticRegression, # L2 Regularized Logistic Regression
                     "XGB": XGBClassifier, # Extreme Gradient Boostring
                     "RF": RandomForestClassifier,
                     "NN": MLPClassifier} # Multilayer perceptron (aka neural network)

        # model calibration parameters
        self.calib_param = {'method': 'isotonic', 'cv': 3}
        self.calib_param_logistic = {'method': 'sigmoid', 'cv': 3}

        # hyperparameter tuning config
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
                                            {'learning_rate_init': (0.001, 0.1),
                                             'batch_size': (64, 512),
                                             'momentum': (0,1),
                                             'alpha': (0,1),
                                             'first_layer_size': (64, 256),
                                             'second_layer_size': (64, 256),
                                             'third_layer_size': (64, 256)})}

    def plot_PR_curve_NN(self, algorithm, model, X, Y, ax):
        """plot Precision-Recall curve for Neural Network
        """
        pred = model.predict_proba(X)
        avg_precision_scores = []
        for i, blood_type in enumerate(self.blood_types):
            col = Y.columns[Y.columns.str.contains(blood_type)]
            y_true = Y[col]
            y_scores = pred[:, i]
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            avg_precision_scores.append(np.round(average_precision_score(y_true, y_scores), 2))
            ax.plot(precision, recall)
        ax.legend([blood_type + f'(AP = {avg_precision_scores[i]})' 
                   for i, blood_type in enumerate(self.blood_types)], loc='lower left')
        ax.set_xlabel('Recall (Positive label: True)')
        ax.set_ylabel('Precision (Positive label: True)')
        plt.title(algorithm)

    def plot_PR_curve(self):
        """plot Precision-Recall curve for all models, write plots to file
        """
        fig = plt.figure(figsize=(12,9))
        plt.subplots_adjust(hspace=0.3)
        for idx, (algorithm, _) in enumerate(self.ml_models.items()):
            # Load model
            filename = f'models/{algorithm}_classifier.pkl'
            with open(filename, 'rb') as file:
                model = pickle.load(file)
            
            ax = fig.add_subplot(2, 2, idx+1)
            if algorithm == 'NN':
                self.plot_PR_curve_NN(algorithm, model, self.X_test, self.Y_test, ax)
                continue
            for idx, blood_type in enumerate(self.blood_types):
                col = self.Y_test.columns[self.Y_test.columns.str.contains(blood_type)]
                plot_precision_recall_curve(model.estimators_[idx], self.X_test, self.Y_test[col], 
                                                name=blood_type, ax=ax)
                plt.title(algorithm)
        plt.savefig('models/pr_curve.jpg')

    def plot_ROC_curve_NN(self, algorithm, model, X, Y, ax):
        """plot Reciever Operating Characteristic curve for Neural Network
        """
        pred = model.predict_proba(X)
        auc_scores = []
        for i, blood_type in enumerate(self.blood_types):
            col = Y.columns[Y.columns.str.contains(blood_type)]
            y_true = Y[col]
            y_scores = pred[:, i]
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            auc_scores.append(np.round(roc_auc_score(y_true, y_scores), 2))
            ax.plot(fpr, tpr)
        ax.legend([blood_type + f'(AUC = {auc_scores[i]})' 
                   for i, blood_type in enumerate(self.blood_types)], loc='lower right')
        ax.set_xlabel('False Positive Rate (Positive label: True)')
        ax.set_ylabel('True Positive Rate (Positive label: True)')
        plt.title(algorithm)

    def plot_ROC_curve(self, algorithm, model, X, Y, ax):
        """plot Reciever Operating Characteristic curve for all models, write plots to file
        """
        fig = plt.figure(figsize=(12,9))
        plt.subplots_adjust(hspace=0.3)
        for idx, (algorithm, _) in enumerate(self.ml_models.items()):
            # Load model
            filename = f'models/{algorithm}_classifier.pkl'
            with open(filename, 'rb') as file:
                model = pickle.load(file)
            
            ax = fig.add_subplot(2, 2, idx+1)
            if algorithm == 'NN':  
                self.plot_ROC_curve_NN(algorithm, model, self.X_test, self.Y_test, ax)
                continue
            for idx, blood_type in enumerate(self.blood_types):
                col = self.Y_test.columns[self.Y_test.columns.str.contains(blood_type)]
                plot_roc_curve(model.estimators_[idx], self.X_test, self.Y_test[col], 
                                    name=blood_type, ax=ax)
                plt.title(algorithm)
        plt.savefig('models/roc_curve.jpg')

    def plot_calibration_curve(self):
        fig = plt.figure(figsize=(12,9))
        for idx, (algorithm, model) in enumerate(self.ml_models.items()):
            # Load model
            filename = f'models/{algorithm}_classifier.pkl'
            with open(filename, 'rb') as file:
                model = pickle.load(file)
            
            ax = fig.add_subplot(2, 2, idx+1)
            ax.plot([0,1],[0,1],'k:', label='perfect calibration')
            pred_prob = model.predict_proba(self.X_train) 
            for i, blood_type in enumerate(self.blood_types):
                col = self.Y_train.columns[self.Y_train.columns.str.contains(blood_type)]
                y_true = self.Y_train[col]
                y_pred = pred_prob[:, i] if algorithm=='NN' else pred_prob[i][:, 1]
                prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=20)
                ax.plot(prob_true, prob_pred, label=blood_type)
            ax.legend()
            plt.title(algorithm)    
        plt.savefig('models/calibration_curve.jpg')

    def evaluate(self, model, eval_NN=False):
        model.fit(self.X_train, self.Y_train)
        pred_prob = model.predict_proba(self.X_valid)
        result = []
        for i, blood_type in enumerate(self.blood_types):
            col = self.Y_valid.columns[self.Y_valid.columns.str.contains(blood_type)]
            Y_true = self.Y_valid[col]
            Y_pred_prob = pred_prob[:, i] if eval_NN else pred_prob[i][:, 1]
            result.append(average_precision_score(Y_true, Y_pred_prob))
        return np.mean(result)

    def lr_evaluate(self, C):
        params = {'C': C, 
                  'class_weight': 'balanced',
                  'max_iter': 1000,
                  'random_state': 42}
        model = MultiOutputClassifier(CalibratedClassifierCV(self.ml_models['LR'](**params), **self.calib_param_logistic))
        return self.evaluate(model)

    # weight for positive examples to account for imbalanced dataset
    # scale_pos_weight = [neg_count/pos_count for index, (neg_count, pos_count) in Y_distribution['Train'].iterrows()]
    # min_child_weight = max(scale_pos_weight) * 6 # can't have less than 6 samples in a leaf node
    def xgb_evaluate(self, learning_rate, n_estimators, max_depth, gamma, reg_lambda):
        params = {'learning_rate': learning_rate, 
                  'n_estimators': int(n_estimators), 
                  'max_depth': int(max_depth),
                  'gamma': gamma, 
                  'reg_lambda': reg_lambda,
                  # 'scale_pos_weight': scale_pos_weight,
                  # 'min_child_weight': min_child_weight, # set to 6 if not using scale_pos_weight
                  'min_child_weight': 6,
                  'random_state': 42,
                  'n_jobs': min(os.cpu_count(), 32),
                 }
        model = MultiOutputClassifier(CalibratedClassifierCV(self.ml_models['XGB'](**params), **self.calib_param))
        return self.evaluate(model)

    def rf_evaluate(self, n_estimators, max_depth, max_features):
        params = {'n_estimators': int(n_estimators),
                  'max_depth': int(max_depth),
                  'max_features': max_features,
                  # can't allow leaf node to have less than 6 samples in accordance with ICES privacy policies
                  'min_samples_leaf': 6,
                  'class_weight': 'balanced_subsample',
                  'random_state': 42,
                  'n_jobs': min(os.cpu_count(), 32)}
        model = MultiOutputClassifier(CalibratedClassifierCV(self.ml_models['RF'](**params), **self.calib_param))
        return self.evaluate(model)

    def nn_evaluate(self, learning_rate_init, batch_size, momentum, alpha, 
                    first_layer_size, second_layer_size, third_layer_size):
        params = {'learning_rate_init': learning_rate_init,
                  'batch_size': int(batch_size),
                  'momentum': momentum,
                  'alpha': alpha,
                  'hidden_layer_sizes': (int(first_layer_size), int(second_layer_size), int(third_layer_size)),
                  'max_iter': 100,
                  'random_state': 42}
        model = CalibratedClassifierCV(self.ml_models['NN'](**params), **self.calib_param)
        return self.evaluate(model, eval_NN=True)

    def convert_some_params_to_int(self, best_param):
        for param in ['max_depth', 'batch_size', 'n_estimators',
                      'first_layer_size', 'second_layer_size', 'third_layer_size']:
            if param in best_param:
                best_param[param] = int(best_param[param])
        return best_param

    def train_model_with_best_param(self, algorithm, model, best_param):
        if algorithm in ['XGB', 'RF']:
            model = MultiOutputClassifier(model(**best_param))
        elif algorithm == 'NN':
            if not 'hidden_layer_sizes' in best_param:
                best_param['hidden_layer_sizes'] = (best_param['first_layer_size'], 
                                                    best_param['second_layer_size'], 
                                                    best_param['third_layer_size'])
                del best_param['first_layer_size'], best_param['second_layer_size'], best_param['third_layer_size']
            model = model(**best_param)
        elif algorithm == 'LR':
            model = MultiOutputClassifier(model(max_iter=1000, **best_param))
        model.fit(self.X_train, self.Y_train)
        return model

    def baseline_model(self):
        """Evaluate Baseline Model - Predict Previous Value
        """
        for metric in ['Acc', 'Precision', 'Recall', 'F1 Score']:
            self.score_df.loc[('Baseline', metric), :] = np.nan
        for split, X, Y in [('Train', self.X_train, self.Y_train), 
                            ('Valid', self.X_valid, self.Y_valid), 
                            ('Test', self.X_test, self.Y_test)]:
            for blood_type in self.blood_types:
                col = Y.columns[Y.columns.str.contains(blood_type)]
                Y_true = Y[col]
                Y_pred = X[f'prev_{blood_type}_value'] < self.neutrophil_threshold
                report = classification_report(Y_true, Y_pred, output_dict=True)
                
                self.score_df.loc[('Baseline', 'Acc'), (split, blood_type)] = report['accuracy']
                
                # predicted true positive over all predicted positive
                self.score_df.loc[('Baseline', 'Precision'), (split, blood_type)] = report['True']['precision']
                
                # predicted true positive over all true positive (aka senstivity)
                self.score_df.loc[('Baseline', 'Recall'), (split, blood_type)] = report['True']['recall']
                
                # 2*precision*recall / (precision + recall)
                self.score_df.loc[('Baseline', 'F1 Score'), (split, blood_type)] = report['True']['f1-score']

    def bayes_hyperparam_optim(self):
        """Run Bayesian Optimization on all the machine learning models, write best hyperparams to file
        """
        for algorithm, model in self.ml_models.items():
            # if algorithm in ['LR', 'XGB']: continue # put the algorithms already trained and tuned in this list
                
            # Conduct Bayesian Optimization
            evaluate_function, optim_config, hyperparam_config = self.model_tuning_config[algorithm]
            bo = BayesianOptimization(evaluate_function, hyperparam_config)
            bo.maximize(acq='ei', **optim_config)
            best_param = bo.max['params']
            best_param = self.convert_some_params_to_int(best_param)
            print(f'Finished finding best hyperparameters for {algorithm}')
            
            # Save the best hyperparameters
            param_filename = f'models/{algorithm}_classifier_best_param.pkl'
            with open(param_filename, 'wb') as file:    
                pickle.dump(best_param, file)

    def train_and_evaluate_all(self, evaluate_only=False):
        """Train all the machine learning models using best hyperparams, 
        Evaluate them, putting results in score_df
        Create AUROC, PR, and calibration plots
        """
        self.baseline_model()
        for algorithm, model in self.ml_models.items():
            for metric in ['Acc', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score', 'AP Score']:
                score_df.loc[(algorithm, metric), :] = np.nan
            
            if evaluate_only:
                # Load best model
                filename = f'models/{algorithm}_classifier.pkl'
                with open(filename, 'rb') as file:
                    model = pickle.load(file)
            else:
                # Load best params
                filename = f'models/{algorithm}_classifier_best_param.pkl'
                with open(filename, 'rb') as file:
                    best_param = pickle.load(file)
                # Train model with best param
                model = train_model_with_best_param(algorithm, model, best_param)
                # Save the model
                model_filename = f'models/{algorithm}_classifier.pkl'
                with open(model_filename, 'wb') as file:
                    pickle.dump(model, file)
                
            # Evaluate the model
            for split, X, Y in [('Train', self.X_train, self.Y_train), 
                                ('Valid', self.X_valid, self.Y_valid), 
                                ('Test', self.X_test, self.Y_test)]:
                # 3 x n x 2 matrix, first column is prob of false, second column is prob of true
                pred_prob = model.predict_proba(X) 
                for idx, blood_type in enumerate(self.blood_types):
                    col = Y.columns[Y.columns.str.contains(blood_type)]
                    Y_true = Y[col]
                    Y_pred_prob = pred_prob[:, idx] if algorithm=='NN' else pred_prob[idx][:, 1]
                    Y_pred_bool = Y_pred_prob > 0.5
                    report = classification_report(Y_true, Y_pred_bool, output_dict=True, zero_division=1)
                    self.score_df.loc[(algorithm, 'Acc'), (split, blood_type)] = report['accuracy']
                    self.score_df.loc[(algorithm, 'Precision'), (split, blood_type)] = report['True']['precision']
                    self.score_df.loc[(algorithm, 'Recall'), (split, blood_type)] = report['True']['recall']
                    self.score_df.loc[(algorithm, 'F1 Score'), (split, blood_type)] = report['True']['f1-score']
                    self.score_df.loc[(algorithm, 'ROC AUC Score'), (split, blood_type)] = roc_auc_score(Y_true, Y_pred_prob)
                    self.score_df.loc[(algorithm, 'AP Score'), (split, blood_type)] = average_precision_score(Y_true, Y_pred_prob)
                    
                    # display confusion matrix
                    if split == 'Test':
                        cm = confusion_matrix(Y_true, Y_pred_bool)
                        cm = pd.DataFrame(cm, columns=['Predicted False', 'Predicted True'], index=['Actual False', 'Actual True'])
                        print(f"\n######## {algorithm} - {split} - {blood_type} #########")
                        print(cm)

        self.score_df = pd.DataFrame(self.score_df.values.astype(float).round(4), 
                                        index=self.score_df.index, columns=self.score_df.columns)
        self.score_df.to_csv('models/classification_results.csv')
        print(self.score_df)

        self.plot_ROC_curve()
        self.plot_PR_curve()
        self.plot_calibration_curve()

def main():
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--find_best_param', action='store_true', 
        help='Conduct bayesian optimization to find best hyperparameters')
    parser.add_argument('--evaluate_only', action='store_true', 
        help='Already have trained model pickle files. Do not train again.')
    
    train = Train()
    if args.find_best_param:
        train.bayes_hyperparam_optim()
    train.train_and_evaluate_all(self.evaluate_only)

if __name__ == '__main__':
    main()