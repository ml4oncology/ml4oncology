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
import os
import tqdm
import pickle
import pandas as pd
import numpy as np
from functools import partial
from scipy.stats import pearsonr
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import (average_precision_score, precision_score, recall_score, roc_auc_score)
from scripts.config import (root_path, cytopenia_gradings, blood_types, all_observations, 
                            symptom_cols, cancer_code_mapping, nn_solvers, nn_activations, 
                            clean_variable_mapping, variable_groupings_by_keyword, eGFR_params)
from scripts.preprocess import (clean_string, split_and_parallelize)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', datefmt='%I:%M:%S')

twolevel = pd.MultiIndex.from_product([[], []])

# Initalize project result folders
def initialize_folders(output_path, extra_folders=None):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        if extra_folders is None: extra_folders = []
        main_folders = ['confidence_interval', 'perm_importance', 'best_params', 'predictions', 'tables', 'figures']
        figure_folders = ['important_features', 'curves', 'subgroup_performance', 'rnn_train_performance', 'decision_curve']
        figure_folders = [f'figures/{folder}' for folder in figure_folders]
        for folder in main_folders + figure_folders + extra_folders:
            os.makedirs(f'{output_path}/{folder}')

# Load / Save
def load_ml_model(model_dir, algorithm, name='classifier'):
    filename = f'{model_dir}/{algorithm}_{name}.pkl'
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def save_predictions(preds, save_dir, filename='predictions'):
    filename = f'{save_dir}/{filename}.pkl'
    with open(filename, 'wb') as file:    
        pickle.dump(preds, file)

def load_predictions(save_dir, filename='predictions'):
    filename = f'{save_dir}/{filename}.pkl'
    with open(filename, 'rb') as file:
        preds = pickle.load(file)
    return preds

def load_ensemble_weights(save_dir):
    filename = f'{save_dir}/ENS_best_param.pkl'
    with open(filename, 'rb') as file:
        ensemble_weights = pickle.load(file)
    return ensemble_weights

# Bootstrap Confidence Interval 
def bootstrap_worker(bootstrap_partition, Y_true, Y_pred):
    """
    Compute bootstrapped AUROC/AUPRC scores by resampling the labels and predictions together
    and recomputing the AUROC/AUPRC
    """
    scores = []
    for i in bootstrap_partition:
        np.random.seed(i)
        weights = np.random.random(len(Y_true)) 
        y_true = Y_true.sample(n=len(Y_true), replace=True, random_state=i, weights=weights)
        y_pred = Y_pred[y_true.index]
        if y_true.nunique() < 2:
            continue
        scores.append((roc_auc_score(y_true, y_pred), average_precision_score(y_true, y_pred)))
    return scores

def compute_bootstrap_scores(Y_true, Y_pred, n_bootstraps=10000, processes=32):
    worker = partial(bootstrap_worker, Y_true=Y_true, Y_pred=Y_pred)
    scores = split_and_parallelize(range(n_bootstraps), worker, split_by_ikns=False, processes=processes)
    
    n_skipped = n_bootstraps - len(scores)
    if n_skipped > 0: 
        logging.warning(f'{n_skipped} bootstraps with no pos examples were skipped, ' + \
                        f'skipped bootstraps will be replaced with original score')
        # fill skipped boostraps with the original score
        scores += [(roc_auc_score(Y_true, Y_pred), average_precision_score(Y_true, Y_pred)), ] * n_skipped
        
    return scores

def nadir_bootstrap_worker(bootstrap_partition, df, days, thresh):
    """
    Compute bootstrapped nadir days by resampling patients with replacement and recomputing nadir day
    For uncertainty estimation of the actual nadir day
    """
    nadir_days = []
    ikns = df['ikn'].unique()
    for i in bootstrap_partition:
        np.random.seed(i)
        sampled_ikns = np.random.choice(ikns, len(ikns), replace=True)
        sampled_df = df.loc[df['ikn'].isin(sampled_ikns), days]
        cytopenia_rates_per_day = get_cyto_rates(sampled_df, thresh)
        if all(cytopenia_rates_per_day == 0):
            # if no event, pick random day in the cycle
            nadir_day = np.random.choice(days)
        else:
            nadir_day = np.argmax(cytopenia_rates_per_day)
        nadir_days.append(nadir_day+1)
    return nadir_days

# Data Descriptions
def get_nunique_entries(df):
    catcols = df.dtypes[df.dtypes == object].index.tolist()
    return pd.DataFrame(df[catcols].nunique(), columns=['Number of Unique Entries']).T

def get_nmissing(df, verbose=False):
    missing = df.isnull().sum() # number of nans for each column
    missing = missing[missing != 0] # remove columns without missing values
    missing = pd.DataFrame(missing, columns=['Missing (N)'])
    missing['Missing (%)'] = (missing['Missing (N)'] / len(df) * 100).round(3)
    
    if verbose:
        other = ['intent_of_systemic_treatment', 'lhin_cd', 'curr_morph_cd', 'curr_topog_cd', 'body_surface_area']
        idx = missing.index
        mapping = {'lab tests': missing.loc[idx.str.contains('baseline')],
                   'symptom values': missing.loc[idx[idx.isin(symptom_cols)]],
                   'other data': missing.loc[idx[idx.isin(other)]]}
        for name, miss in mapping.items():
            miss_percentage = miss['Missing (%)']
            logging.info(f'{miss_percentage.min()}%-{miss_percentage.max()}% of {name} were missing before treatment sessions')
        
    return missing.sort_values(by='Missing (N)')

def get_nmissing_by_splits(df, labels):
    missing = [get_nmissing(df.loc[Y.index]) for Y in labels.values()]
    missing = pd.concat(missing, axis=1)
    cols_upper = [f'{split} (N={len(Y)})' for split, Y in labels.items()]
    cols_lower = missing.columns.unique().tolist()
    cols =  pd.MultiIndex.from_product([cols_upper, cols_lower])
    missing.columns = cols
    return missing

def get_clean_variable_names(names):
    """
    Args:
        names: array-like of strings (variable names)
    """
    names = pd.Index(names)
    # swap cause and event
    for event in ['ED', 'H']:
        for cause in ['INFX', 'TR', 'GI']:
            names = names.str.replace(f'{cause}_{event}', f'{event}_{cause}')
    # clean_variable_mapping = list(clean_variable_mapping.items()) + list(cancer_code_mapping.items())
    for orig, new in clean_variable_mapping.items():
        names = names.str.replace(orig, new)
    names = names.str.replace('_', ' ')
    names = names.str.title()
    names = names.str.replace('Ed', 'ED')
    return names

def get_observation_units():
    filename = f'{root_path}/data/olis_units.pkl'
    with open(filename, 'rb') as file:
        units_map = pickle.load(file)
    
    # group the observation units together with same observation name
    grouped_units_map = {}
    for obs_code, obs_name in all_observations.items():
        obs_name = get_clean_variable_names([obs_name])[0]
        if obs_name in grouped_units_map:
            grouped_units_map[obs_name].append(units_map[obs_code])
        else:
            grouped_units_map[obs_name] = [units_map[obs_code]]

    # just keep the first observation unit, since they are all the same
    for obs_name, units in grouped_units_map.items():
        assert len(set(units)) == 1
        grouped_units_map[obs_name] = units[0]
        
    return grouped_units_map

def get_cyto_rates(df, thresh):
    """
    Get cytopenia rate on nth day of chemo cycle, based on available measurements
    """
    cytopenia_rates_per_day = []
    for day, measurements in df.iteritems():
        measurements = measurements.dropna()
        cytopenia_rate = (measurements < thresh).mean()
        cytopenia_rates_per_day.append(cytopenia_rate)
    cytopenia_rates_per_day = pd.Index(cytopenia_rates_per_day).fillna(0)
    return cytopenia_rates_per_day 
    
def most_common_by_category(data, category='regimen', with_respect_to='sessions', top=5):
    # most common `category` (e.g. regimen, cancer location, cancer type) with respect to patients or sessions
    if with_respect_to == 'patients':
        get_num_patients = lambda group: group['ikn'].nunique()
        npatients_by_category = data.groupby(category).apply(get_num_patients)
    elif with_respect_to == 'sessions':
        npatients_by_category = data.groupby(category).apply(len)
    else:
        raise ValueError('Must be with respect to patients or sessions')
    most_common = npatients_by_category.sort_values(ascending=False)[0:top]
    most_common = most_common.to_dict()
    return most_common

def top_cancer_regimen_summary(data, top=10):
    result = pd.DataFrame()
    top_cancers = most_common_by_category(data, category='curr_topog_cd', top=top) # cancer locations
    for cancer_code, nsessions in top_cancers.items():
        top_regimens = most_common_by_category(data[data['curr_topog_cd'] == cancer_code], category='regimen', top=top)
        top_regimens = [f'{regimen} ({num_sessions})' for regimen, num_sessions in top_regimens.items()]
        result[f'{cancer_code_mapping[cancer_code]} (N={nsessions})'] = top_regimens
    result.index += 1
    return result

class DataPartitionSummary:
    def __init__(self, X, Y, partition, top_category_items=None):
        self.top_category_items = {} if top_category_items is None else top_category_items
        self.patient_indices = X['ikn'].drop_duplicates(keep='last').index
        p = 'patients'
        s = 'sessions'
        self.total = {p: len(self.patient_indices),
                      s: len(X)}
        self.col = {p: (partition, f'Patients (N={self.total[p]})'),
                    s: (partition, f'Treatment Sessions (N={self.total[s]})')}
        self.X = X
        self.Y = Y.loc[X.index]
        self.display = lambda x, total: f"{x} ({np.round(x/total*100, 1)}%)"
        
    def num_sessions_per_patient_summary(self, summary_df):
        num_sessions = self.X.groupby('ikn').apply(len)
        mean = np.round(num_sessions.mean(), 1)
        std = np.round(num_sessions.std(), 1)
        summary_df.loc[('Number of Sessions', ''), self.col['patients']] = f'{mean}, SD ({std})'

    def median_age_summary(self, summary_df):
        for with_respect_to, col in self.col.items():
            if with_respect_to == 'patients':
                age = self.X.loc[self.patient_indices, 'age']
            elif with_respect_to == 'sessions':
                age = self.X['age']
            q25, q75 = np.percentile(age, [25, 75]).astype(int)
            summary_df.loc[('Median Age', ''), col] = f"{int(age.median())}, IQR [{q25}-{q75}]"

    def sex_summary(self, summary_df):
        for with_respect_to, col in self.col.items():
            total = self.total[with_respect_to]
            if with_respect_to == 'patients':
                F, M = self.X.loc[self.patient_indices, 'sex'].value_counts()
            elif with_respect_to == 'sessions':
                F, M = self.X['sex'].value_counts()
            summary_df.loc[('Sex', 'Female'), col] = self.display(F, total)
            summary_df.loc[('Sex', 'Male'), col] = self.display(M, total)
            
    def immigration_summary(self, summary_df):
        for with_respect_to, col in self.col.items():
            total = self.total[with_respect_to]
            if with_respect_to == 'patients':
                num_non_immigrants, num_immigrants = self.X.loc[self.patient_indices, 'is_immigrant'].value_counts()
                num_eng_speakers, num_non_eng_speakers = self.X.loc[self.patient_indices, 'speaks_english'].value_counts()
            elif with_respect_to == 'sessions':
                num_nonimmigrants, num_immigrants = self.X['is_immigrant'].value_counts()
                num_eng_speakers, num_non_eng_speakers = self.X['speaks_english'].value_counts()
            summary_df.loc[('Immigration', 'Immigrant'), col] = self.display(num_immigrants, total)
            summary_df.loc[('Immigration', 'Non-immigrant'), col] = self.display(num_non_immigrants, total)
            summary_df.loc[('Immigration', 'English Speaker'), col] = self.display(num_eng_speakers, total)
            summary_df.loc[('Immigration', 'Non-English Speaker'), col] = self.display(num_non_eng_speakers, total)
            
    def other_summary(self, summary_df, col, display_name, top_items):
        """summarize items not in the top_items
        """
        other = self.X[~self.X[col].isin(top_items.keys())]
        num_patients = other['ikn'].nunique()
        num_sessions = len(other)
        for num, wrt in [(num_patients, 'patients'), (num_sessions, 'sessions')]:
            summary_df.loc[(display_name, 'Other'), self.col[wrt]] = self.display(num, self.total[wrt])

    def category_subgroup_summary(self, summary_df, category='regimen', top=5):
        display_name = get_clean_variable_names([category])[0]
        
        if category not in self.top_category_items:
            top_items = most_common_by_category(self.X, category=category, top=top)
            self.top_category_items[category] = list(top_items)
        else:
            top_items = {item: sum(self.X[category] == item) for item in self.top_category_items[category]}
        
        for item, num_sessions in top_items.items():
            num_patients = self.X.loc[self.X[category] == item, 'ikn'].nunique()
            if category in ['curr_topog_cd', 'curr_morph_cd']: item = cancer_code_mapping[item]
            row = (display_name, item)
            summary_df.loc[row, self.col['patients']] = self.display(num_patients, self.total['patients'])
            summary_df.loc[row, self.col['sessions']] = self.display(num_sessions, self.total['sessions'])
        # summarize the regimens not in top_regimens
        self.other_summary(summary_df, category, display_name, top_items)
        
    def combordity_summary(self, summary_df):
        for with_respect_to, col in self.col.items():
            total = self.total[with_respect_to]
            if with_respect_to == 'patients':
                num_non_diabetes, num_diabetes = self.X.loc[self.patient_indices, 'diabetes'].value_counts()
                num_non_ht, num_ht = self.X.loc[self.patient_indices, 'hypertension'].value_counts()
            elif with_respect_to == 'sessions':
                num_non_diabetes, num_diabetes = self.X['diabetes'].value_counts()
                num_non_ht, num_ht = self.X['hypertension'].value_counts()
            summary_df.loc[('Combordity', 'Diabetes'), col] = self.display(num_diabetes, total)
            summary_df.loc[('Combordity', 'Non-diabetes'), col] = self.display(num_non_diabetes, total)
            summary_df.loc[('Combordity', 'Hypertension'), col] = self.display(num_ht, total)
            summary_df.loc[('Combordity', 'Non-hypertension'), col] = self.display(num_non_ht, total)
    
    def gcsf_summary(self, summary_df):
        X = self.X[self.X['age'] >= 65] # summary for only among those over 65
        for with_respect_to, col in self.col.items():
            if with_respect_to == 'patients':
                total = X['ikn'].nunique()
                num_gcsf_given = X.loc[X['ODBGF_given'], 'ikn'].nunique()
            elif with_respect_to == 'sessions':
                total = len(X)
                num_gcsf_given = X['ODBGF_given'].sum()
            summary_df.loc[('GCSF Administered', ''), col] = self.display(num_gcsf_given, total)
            
    def event_summary(self, mask, row, summary_df):
        for with_respect_to, col in self.col.items():
            total = self.total[with_respect_to]
            if with_respect_to == 'patients':
                num_events = self.X.loc[mask, 'ikn'].nunique()
            elif with_respect_to == 'sessions':
                num_events = mask.sum()
            summary_df.loc[row, col] = self.display(num_events, total)
            
    def ckd_summary(self, summary_df):
        mask = self.X['baseline_eGFR'] < 60
        row = ('CKD prior to treatment', '')
        self.event_summary(mask, row, summary_df)
    
    def dialysis_summary(self, summary_df):
        mask = self.X['dialysis']
        row = ('Dialysis after treatment', '')
        self.event_summary(mask, row, summary_df)
            
    def target_summary(self, summary_df):
        for target, Y in self.Y.iteritems():
            row = ('Target Event', target)
            self.event_summary(Y, row, summary_df)
    
    def get_summary(self, summary_df, top=3, include_target=True, include_combordity=False, 
                    include_gcsf=False, include_ckd=False, include_dialysis=False):
        self.num_sessions_per_patient_summary(summary_df)
        self.median_age_summary(summary_df)
        self.immigration_summary(summary_df)
        self.sex_summary(summary_df)
        self.category_subgroup_summary(summary_df, category='regimen', top=top)
        self.category_subgroup_summary(summary_df, category='curr_topog_cd', top=top)
        self.category_subgroup_summary(summary_df, category='curr_morph_cd', top=top)
        if include_target: self.target_summary(summary_df)
        if include_combordity: self.combordity_summary(summary_df)
        if include_gcsf: self.gcsf_summary(summary_df)
        if include_ckd: self.ckd_summary(summary_df)
        if include_dialysis: self.dialysis_summary(summary_df)

def data_characteristic_summary(eval_models, save_dir, partition='split', **kwargs):
    """
    Get characteristics summary of patients and treatments for each 
    data split (Train-Valid-Test) or cohort (Development-Testing)
    
    Development cohort refers to Training and Validation split
    
    Args:
        partition (str): how to partition the data for summarization, either by split or cohort
    """
    model_data = eval_models.orig_data
    labels = eval_models.labels
    summary_df = pd.DataFrame(index=twolevel, columns=twolevel)
    
    # Data full summary
    Y = pd.concat(labels.values())
    X = model_data.loc[Y.index]
    dps = DataPartitionSummary(X, Y, 'All')
    dps.get_summary(summary_df, **kwargs)
    top_category_items = dps.top_category_items
    
    # Data partition summary
    if partition == 'split':
        # Train, Valid, Test data splits
        groupings = labels.items()
    elif partition == 'cohort':
        # Development, Testing cohort
        groupings = [('Development', pd.concat([labels['Train'], labels['Valid']])),
                     ('Testing', labels['Test'])]
    for partition_name, Y in groupings:
        X = model_data.loc[Y.index]
        dps = DataPartitionSummary(X, Y, partition_name, top_category_items=top_category_items)
        dps.get_summary(summary_df, **kwargs)
        
    summary_df.to_csv(f'{save_dir}/data_characteristic_summary.csv')
    return summary_df

def feature_summary(eval_models, prep, target_keyword, save_dir):
    df = prep.dummify_data(eval_models.orig_data.copy())
    train_idxs = eval_models.labels['Train'].index

    # remove missingness features, targets, first visit date, and ikn
    cols = df.columns
    df = df[cols[~(cols.str.contains('is_missing') | 
                   cols.str.contains(target_keyword) | 
                   cols.str.contains('first_visit_date'))].drop('ikn')]

    # get mean, SD, and number of missing values for each feature for the training set
    summary = df.loc[train_idxs].astype(float).describe()
    summary = summary.loc[['count', 'mean', 'std']].T
    summary = summary.round(6)
    summary['count'] = len(train_idxs) - summary['count']
    summary = summary.rename(columns={'count': 'Missingness_training', 'mean': 'Mean_training', 'std': 'SD_training'})

    # assign the groupings for each feature
    features = summary.index
    for group, keyword in variable_groupings_by_keyword.items():
        summary.loc[features.str.contains(keyword), 'Group'] = group
    
    summary.index = get_clean_variable_names(summary.index)
    # insert observation units
    rename_map = {name: f'{name} ({unit})' for name, unit in get_observation_units().items()}
    summary = summary.rename(index=rename_map)
    
    summary.to_csv(f'{save_dir}/feature_summary.csv')
    return summary

def nadir_summary(df, output_path, cytopenia='Neutropenia', load_ci=False, n_bootstraps=1000, processes=32):
    """
    Args:
        load_ci: loads the bootstrapped nadir days to compute nadir day confidence interval
    """
    if cytopenia not in {'Neutropenia', 'Anemia', 'Thrombocytopenia'}: 
        raise ValueError('cytopenia must be one of Neutropneia, Anemia, or Thrombocytopenia')
        
    if load_ci:
        ci_df = pd.read_csv(f'{output_path}/data/analysis/nadir_{cytopenia}_bootstraps.csv') 
        ci_df = ci_df.set_index('regimen')
    else:
        ci_df = pd.DataFrame()
        
    cycle_lengths = dict(df[['regimen', 'cycle_length']].values)
    result = {}
    for regimen, group in tqdm.tqdm(df.groupby('regimen')):
        cycle_length = int(cycle_lengths[regimen])
        days = range(0, cycle_length)
        result[regimen] = {'NSessions': len(group), 'Cycle Length': cycle_length}
        for grade, thresholds in cytopenia_gradings.items():
            if cytopenia not in thresholds: continue
            thresh = thresholds[cytopenia]   
            cytopenia_rates_per_day = get_cyto_rates(group[days], thresh)
            if all(cytopenia_rates_per_day == 0):
                # if no cytopenia was observed for all days
                worst_cytopenia_rate = 0
                ci_lower, ci_upper = 0, 1
            else:
                nadir_day = np.argmax(cytopenia_rates_per_day)
                nadir_day_measurements = group[nadir_day].dropna()
                nadir_day_n_events = (nadir_day_measurements < thresh).sum()
                if nadir_day_n_events < 5: 
                    # can't allow small cells less than 5 according to ICES privacy policy
                    worst_cytopenia_rate = 0
                    ci_lower, ci_upper = 0, 1
                else:
                    worst_cytopenia_rate = cytopenia_rates_per_day[nadir_day].round(3)

                    # binomial confidence interval for cytopenia rate
                    # since we are working with binomial distribution (i.e. cytopenia - 1, not cytopenia - 0)
                    ci_lower, ci_upper = proportion_confint(count=nadir_day_n_events, 
                                                            nobs=len(nadir_day_measurements), 
                                                            method='wilson',
                                                            alpha=(1-0.95)) # 95% CI
                    ci_lower, ci_upper = ci_lower.round(3), ci_upper.round(3)

                    if grade == 'Grade 2':
                        # get 95% confidence interval for nadir day using bootstrap technique
                        if regimen not in ci_df.index:
                            bootstraps = range(n_bootstraps)
                            worker = partial(nadir_bootstrap_worker, df=group, days=days, thresh=thresh)
                            ci_df.loc[regimen, bootstraps] = split_and_parallelize(bootstraps, worker, split_by_ikns=False, processes=processes)
                        nadir_days = ci_df.loc[regimen].values
                        nadir_lower, nadir_upper = np.percentile(nadir_days, [2.5, 97.5]).astype(int)

                        # set day 1 as day of administration (not day 0)
                        result[regimen]['Nadir Day'] = f'{nadir_day+1} ({nadir_lower}-{nadir_upper})'
                        # result[regimen]['NMeasurements at Nadir Day'] = len(nadir_day_measurements)
    
            result[regimen][f'{grade} {cytopenia} Rate (<{thresh})'] = f'{worst_cytopenia_rate} ({ci_lower}-{ci_upper})'
        
    summary_df = pd.DataFrame(result).T
    summary_df.to_csv(f'{output_path}/data/analysis/nadir_{cytopenia}_summary.csv')
    ci_df.to_csv(f'{output_path}/data/analysis/nadir_{cytopenia}_bootstraps.csv', index_label='regimen')
    return summary_df
        
# Post Training Analysis
def group_pred_by_outcome(df, event='ACU'):
    """
    Group the predictions by outcome, collapse multiple chemo treatments into one outcome

    E.g. if chemo happens on Jan 1 and Jan 14, ED visit happens on Jan 20, and our lookahead window is 30 days, 
         then the outcome for predicting ED visit within 30 days would be a true positive if either Jan 1 and Jan 14 
         trigger a warning and a false negative if neither do

    Currrently only supports ED/H event and Death outcomes
    """
    result = {} # index: prediction
    if event == 'ACU':
        event_cols = ['next_ED_date', 'next_H_date']
    elif event == 'ED' or event == 'H':
        event_cols = [f'next_{event}_date']
    else:
        event_cols = ['D_date']
    for ikn, ikn_group in df.groupby('ikn'):
        for event in event_cols:
            for date, date_group in ikn_group.groupby(event):
                idx = date_group.index[-1]
                result[idx] = result.get(idx, False) or any(date_group['pred'])
    return list(result.items())
    
def pred_thresh_binary_search(Y_pred_prob, Y_true, desired_target, metric='precision'):
    increase_thresh = lambda low_thresh, mid_thresh, high_thresh: (mid_thresh + 0.0001, high_thresh)
    decrease_thresh = lambda low_thresh, mid_thresh, high_thresh: (low_thresh, mid_thresh - 0.0001)
    if metric == 'precision': # goes up as pred thresh goes up
        score_func = precision_score
        less_than_target_adjust_thresh, more_than_target_adjust_thresh = increase_thresh, decrease_thresh
    elif metric == 'sensitivity': # goes down as pred thresh goes up
        score_func = recall_score
        less_than_target_adjust_thresh, more_than_target_adjust_thresh = decrease_thresh, increase_thresh

    cur_target = 0
    low_thresh, high_thresh = 0, 1
    while low_thresh <= high_thresh:
        mid_thresh = (high_thresh+low_thresh)/2
        Y_pred_bool = Y_pred_prob > mid_thresh
        cur_target = score_func(Y_true, Y_pred_bool, zero_division=1)
        if abs(cur_target - desired_target) <= 0.005:
            return mid_thresh
        elif cur_target < desired_target:
            low_thresh, high_thresh = less_than_target_adjust_thresh(low_thresh, mid_thresh, high_thresh)
        elif cur_target > desired_target:
            low_thresh, high_thresh = more_than_target_adjust_thresh(low_thresh, mid_thresh, high_thresh)
    logging.warning(f'Desired {metric} {desired_target} could not be achieved. Closest {metric} achieved was {cur_target}')
    return mid_thresh

class SubgroupPerformanceSummary:
    def __init__(self, algorithm, eval_models, target_types,
                 pred_thresh=0.2, split='Test', display_ci=False, load_ci=False):
        self.model_dir = eval_models.output_path
        self.algorithm = algorithm
        self.split = split
        self.Y = eval_models.labels[split]
        self.entire_data = eval_models.orig_data # used for getting most common category subgroups
        self.data = self.entire_data.loc[self.Y.index]
        self.N = self.data['ikn'].nunique()
        self.target_types = target_types
        self.preds = eval_models.preds[split][algorithm]
        self.pred_thresh = pred_thresh
        self.display_ci = display_ci
        self.load_ci = load_ci
        self.ci_df = pd.DataFrame(index=twolevel)
        if self.load_ci:
            self.ci_df = pd.read_csv(f'{self.model_dir}/confidence_interval/bootstrapped_subgroup_scores_{algorithm}.csv', index_col=[0,1])
            self.ci_df.columns = self.ci_df.columns.astype(int)
    
    def get_bootstrapped_scores(self, Y_true, Y_pred, row_name, target_type, n_bootstraps=10000):
        group_name, subgroup_name = row_name
        group_name = group_name.lower().replace(' ', '_')
        subgroup_name = subgroup_name.split('(')[0].strip().lower().replace(' ', '_').replace('/', '_') # WARNING: SUPER HARD CODED
        ci_index = f'{group_name}_{subgroup_name}_{target_type}'
        if ci_index not in self.ci_df.index:
            auc_scores = compute_bootstrap_scores(Y_true.reset_index(drop=True), Y_pred, n_bootstraps=n_bootstraps)
            logging.info(f'Completed bootstrap computations for {ci_index}')
            auroc_scores, auprc_scores = np.array(auc_scores).T
            self.ci_df.loc[(ci_index, 'AUROC'), range(n_bootstraps)] = auroc_scores
            self.ci_df.loc[(ci_index, 'AUPRC'), range(n_bootstraps)] = auprc_scores
        return self.ci_df.loc[ci_index].values
        
    def get_confidence_interval(self, Y_true, Y_pred, summary_df, row_name, target_type):
        auroc_scores, auprc_scores = self.get_bootstrapped_scores(Y_true, Y_pred, row_name, target_type)
        for name, scores in [('AUROC', auroc_scores), ('AUPRC', auprc_scores)]:
            lower, upper = np.percentile(scores, [2.5, 97.5]).round(3)
            col_name = (target_type, name)
            summary_df.loc[row_name, col_name] = f'{summary_df.loc[row_name, col_name]} ({lower}-{upper})'
        return summary_df

    def score_within_subgroups(self, Y, row_name, summary_df):
        pred_prob = self.preds.loc[Y.index]
        for target_type in self.target_types:
            Y_true = Y[target_type]
            Y_pred_prob = pred_prob[target_type]
            Y_pred_bool = Y_pred_prob > self.pred_thresh
            if Y_true.nunique() < 2:
                logging.warning(f'No pos examples, skipping {target_type} - {row_name}')
                continue
            
            # AUROC/AUPRC
            summary_df.loc[row_name, (target_type, 'AUROC')] = np.round(roc_auc_score(Y_true, Y_pred_prob), 3)
            summary_df.loc[row_name, (target_type, 'AUPRC')] = np.round(average_precision_score(Y_true, Y_pred_prob), 3)
            if self.display_ci:
                summary_df = self.get_confidence_interval(Y_true, Y_pred_prob, summary_df, row_name, target_type)
                
            # PPV/Sensitivity
            summary_df.loc[row_name, (target_type, 'PPV')] = np.round(precision_score(Y_true, Y_pred_bool, zero_division=1), 3)
            summary_df.loc[row_name, (target_type, 'Sensitivity')] = np.round(recall_score(Y_true, Y_pred_bool, zero_division=1), 3)
            
            # Warning Rate/Event Rate
            # summary_df.loc[row_name, (target_type, 'Warning Rate')] = np.round(Y_pred_bool.mean(), 3)
            summary_df.loc[row_name, (target_type, 'Event Rate')] = np.round(Y_true.mean(), 3)
                
    def score_for_most_common_category_subgroups(self, title, category, summary_df, mapping=None):
        for cat_feature, num_sessions in most_common_by_category(self.entire_data, category=category, top=3).items():
            mask = self.data[category] == cat_feature
            Y_subgroup = self.Y[mask] 
            num_patients = self.data.loc[mask, 'ikn'].nunique()
            if mapping: cat_feature = mapping[cat_feature]
            name = (title, f'{cat_feature} ({np.round(num_patients/self.N*100, 1)}%)')
            self.score_within_subgroups(Y_subgroup, name, summary_df)

    def score_for_days_since_starting_subgroups(self, summary_df):
        col = 'days_since_starting_chemo'
        for (low, high) in [(0, 30), (31, 90), (91, np.inf)]:
            mask = self.data[col].between(low, high)
            Y_subgroup = self.Y[mask]
            interval = f'{low}+' if high == np.inf else f'{low}-{high}'
            name = ('Days Since Starting Regimen', interval)
            self.score_within_subgroups(Y_subgroup, name, summary_df)

    def score_for_entire_test_set(self, summary_df):
        name = (f'Entire {self.split} Cohort', f'{self.N} patients (100%)')
        self.score_within_subgroups(self.Y, name, summary_df)
    
    def score_for_age_subgroups(self, summary_df):
        col = 'age'
        for (low, high) in [(18, 64), (65, np.inf)]:
            mask = self.data[col].between(low, high)
            Y_subgroup = self.Y[mask]
            num_patients = self.data.loc[mask, 'ikn'].nunique()
            interval = f'{low}+' if high == np.inf else f'{low}-{high}'
            name = ('Age',  f'{interval} ({np.round(num_patients/self.N*100, 1)}%)')
            self.score_within_subgroups(Y_subgroup, name, summary_df)
            
    def score_for_sex_subgroups(self, summary_df):
        col = 'sex'
        for sex, name in [('F', 'Female'), ('M', 'Male')]:
            mask = self.data[col] == sex
            Y_subgroup = self.Y[mask]
            num_patients = self.data.loc[mask, 'ikn'].nunique()
            name = ('Sex',  f'{name} ({np.round(num_patients/self.N*100, 1)}%)')
            self.score_within_subgroups(Y_subgroup, name, summary_df)
            
    def score_for_immigrant_subgroups(self, summary_df):
        for mask_bool, col_name, col in [(True, 'Immigrant', 'is_immigrant'), 
                                         (False, 'Non-immigrant', 'is_immigrant'),
                                         (True, 'English Speaker', 'speaks_english'),
                                         (False, 'Non-English Speaker', 'speaks_english')]:
            mask = self.data[col] if mask_bool else ~self.data[col]
            Y_subgroup = self.Y[mask]
            num_patients = self.data.loc[mask, 'ikn'].nunique()
            col = col.replace('_', ' ').title()
            name = ('Immigration',  f'{col_name} ({np.round(num_patients/self.N*100, 1)}%)')
            self.score_within_subgroups(Y_subgroup, name, summary_df)
    
    def score_for_cycle_length_subgroups(self, summary_df):
        for cycle_length, group in self.data.groupby('cycle_length'):
            Y_subgroup = self.Y.loc[group.index] 
            name = ('Cycle Length',  f'{cycle_length}')
            self.score_within_subgroups(Y_subgroup, name, summary_df)
            
    def score_for_ckd_subgroups(self, summary_df):
        mask = self.data['baseline_eGFR'] < 60
        for ckd_presence, col_name in [(True, 'Y'), (False, 'N')]:
            ckd_mask = mask if ckd_presence else ~mask
            Y_subgroup = self.Y[ckd_mask]
            name = ('CKD Prior to Treatment',  f'{col_name}')
            self.score_within_subgroups(Y_subgroup, name, summary_df)
            
    def get_summary(self, subgroups, summary_df):
        if 'all' in subgroups: self.score_for_entire_test_set(summary_df)
        if 'age' in subgroups: self.score_for_age_subgroups(summary_df)
        if 'sex' in subgroups: self.score_for_sex_subgroups(summary_df)
        if 'immigrant' in subgroups: self.score_for_immigrant_subgroups(summary_df)
        if 'regimen' in subgroups: self.score_for_most_common_category_subgroups('Regimen', 'regimen', summary_df)
        if 'cancer_location' in subgroups: 
            self.score_for_most_common_category_subgroups('Cancer Location', 'curr_topog_cd', summary_df, mapping=cancer_code_mapping)
        if 'days_since_starting' in subgroups: self.score_for_days_since_starting_subgroups(summary_df)
        if 'cycle_length' in subgroups: self.score_for_cycle_length_subgroups(summary_df)
        if 'ckd' in subgroups: self.score_for_ckd_subgroups(summary_df)
            
def subgroup_performance_summary(algorithm, eval_models, pred_thresh=0.2, subgroups=None, target_types=None, 
                                 display_ci=False, load_ci=False, save_ci=False):
    """
    Args:
        display_ci (bool): display confidence interval
        load_ci (bool): load saved scores for computing confidence interval or recomuputing the bootstrapped scores
    """
    if target_types is None:
        target_types = eval_models.target_types
        
    if subgroups is None:
        subgroups = {'all', 'age', 'sex', 'immigrant', 'regimen', 'cancer_location', 'days_since_starting'}
        
    summary_df = pd.DataFrame(index=twolevel, columns=twolevel)
    sps = SubgroupPerformanceSummary(algorithm, eval_models, target_types, 
                                     pred_thresh=pred_thresh, display_ci=display_ci, load_ci=load_ci)
    sps.get_summary(subgroups, summary_df)
    
    save_dir = eval_models.output_path
    if save_ci: 
        sps.ci_df.to_csv(f'{save_dir}/confidence_interval/bootstrapped_subgroup_scores_{algorithm}.csv')
    summary_df.to_csv(f'{save_dir}/tables/subgroup_performance_summary_{algorithm}.csv')
    return summary_df

def get_worst_performing_subgroup(eval_models, category='regimen', algorithm='XGB', target_type='Neutropenia', split='Valid'):
    """analyze subgroups with the worst performance (usually in the validation set)"""
    summary_df = pd.DataFrame(columns=twolevel)
    sps = SubgroupPerformanceSummary(algorithm, eval_models, target_types=[target_type], split=split)
    for regimen, group in sps.data.groupby(category):
        Y_subgroup = sps.Y.loc[group.index]
        sps.score_within_subgroups(Y_subgroup, regimen, summary_df)
    summary_df[(target_type, 'NSessions')] = sps.data[category].value_counts()
    summary_df = summary_df.sort_values(by=(target_type, 'AUROC'))
    return summary_df

# get estimated glomerular filteration rate
def get_eGFR(df, col='value', prefix=''):
    """
    Estimate the glomerular filteration rate according to [1] for determining chronic kidney disease

    Reference:
        [1] https://www.kidney.org/professionals/kdoqi/gfr_calculator/formula
    """
    for sex, group in df.groupby('sex'):
        params = eGFR_params[sex]
        scr_count = group[col]*0.0113 # convert umol/L to mg/dL (1 umol/L = 0.0113 mg/dl)
        scr_term = scr_count/params['K']
        min_scr_term, max_scr_term = scr_term.copy(), scr_term.copy()
        min_scr_term[min_scr_term > 1] = 1
        max_scr_term[max_scr_term < 1] = 1
        eGFR = 142*(min_scr_term**params['a'])*(max_scr_term**-1.2)*(0.9938**group['age'])*params['multiplier']
        df.loc[group.index, f'{prefix}eGFR'] = eGFR
    return df

# Get hyperparams
def get_hyperparameters(output_path, days=None, algorithms=None):
    if algorithms is None: 
        algorithms = ['LR', 'RF', 'XGB', 'NN', 'ENS', 'RNN']
        
    hyperparams = pd.DataFrame(index=twolevel, columns=['Hyperparameter Value'])
    for algorithm in algorithms:
        filepath = f'{output_path}/best_params/{algorithm}_best_param.pkl'
        with open(filepath, 'rb') as file:
            best_param = pickle.load(file)
        for param, value in best_param.items():
            if param == 'solver': value = nn_solvers[int(value)].upper()
            if param == 'activation': value = nn_activations[int(value)].upper()
            param = param.replace('_', ' ').title()
            if algorithm == 'ENS': param = f'{param.upper()} Weight'
            hyperparams.loc[(algorithm, param), 'Hyperparameter Value'] = value
    hyperparams.to_csv(f'{output_path}/tables/hyperparameters.csv', index_label='index')
    return hyperparams

# Pearson Correlation
def get_pearson_matrix(df, target_keyword, output_path):
    dtypes = df.dtypes
    cols = dtypes[~(dtypes == object)].index
    cols = cols.drop('ikn')
    target_cols = cols[cols.str.contains(target_keyword)]
    feature_cols = cols[~cols.str.contains(target_keyword)]
    
    pearson_matrix = pd.DataFrame(columns=feature_cols, index=target_cols)
    for target in target_cols:
        for feature in tqdm.tqdm(feature_cols):
            data = df[~df[feature].isnull()]
            corr, prob = pearsonr(data[target], data[feature])
            pearson_matrix.loc[target, feature] = np.round(corr, 3)
    pearson_matrix = pearson_matrix.T
    pearson_matrix.index = get_clean_variable_names(pearson_matrix.index)
    pearson_matrix.columns = pearson_matrix.columns.str.replace(target_keyword, '')
    pearson_matrix.to_csv(f'{output_path}/tables/pearson_matrix.csv', index_label='index')
    return pearson_matrix
