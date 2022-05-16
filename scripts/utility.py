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
import pickle
import tqdm
import pandas as pd
import numpy as np
from functools import partial
from scipy.stats import pearsonr
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
        figure_folders = ['figures/important_features', 'figures/curves', 'figures/subgroup_performance', 'figures/rnn_train_performance']
        for folder in main_folders + figure_folders + extra_folders:
            os.makedirs(f'{output_path}/{folder}')
            
# Calculate the Min/Max Expected (Reference Range) Observation Values
def get_observation_ranges():
    obs_ranges = {obs_code: [np.inf, 0] for obs_code in set(all_observation.values())}
    cols = ['ReferenceRange', 'ObservationCode']
    chunks = pd.read_csv(f"{root_path}/data/olis_complete.csv", chunksize=10**6)
    for i, chunk in tqdm.tqdm(enumerate(chunks)):
        # keep columns of interest
        chunk = chunk[cols].copy()
        # remove rows where no reference range is given
        chunk = chunk[chunk['ReferenceRange'].notnull()]
        # clean up string values
        chunk = clean_string(chunk, cols)
        # map the blood type to blood code
        chunk['ObservationCode'] = chunk['ObservationCode'].map(all_observations)
        # get the min/max blood count values for this chunk and update the global min/max blood range
        for observation_code, group in chunk.groupby('ObservationCode'):
            ranges = group['ReferenceRange'].str.split('-')
            min_count = min(ranges.str[0].replace(r'^\s*$', np.nan, regex=True).fillna('inf').astype(float))
            max_count = max(ranges.str[1].replace(r'^\s*$', np.nan, regex=True).fillna('0').astype(float))
            obs_ranges[obs_type][0] = min(min_count, obs_ranges[obs_type][0])
            obs_ranges[obs_type][1] = max(max_count, obs_ranges[obs_type][1])
    return obs_ranges

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
    filename = f'{save_dir}/ENS_classifier_best_param.pkl'
    with open(filename, 'rb') as file:
        ensemble_weights = pickle.load(file)
    return ensemble_weights

# Bootstrap Confidence Interval 
def bootstrap_worker(bootstrap_partition, Y_true, Y_pred):
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

def get_clean_variable_names(cols):
    # swap cause and event
    for event in ['ED', 'H']:
        for cause in ['INFX', 'TR', 'GI']:
            cols = cols.str.replace(f'{cause}_{event}', f'{event}_{cause}')
    # clean_variable_mapping = list(clean_variable_mapping.items()) + list(cancer_code_mapping.items())
    for orig, new in clean_variable_mapping.items():
        cols = cols.str.replace(orig, new)
    cols = cols.str.replace('_', ' ')
    cols = cols.str.title()
    cols = cols.str.replace('Ed', 'ED')
    return cols
    
def most_common_by_category(data, category='regimen', top=5):
    # most common `category` (e.g. regimen, cancer location, cancer type) in terms of number of patients
    get_num_patients = lambda group: group['ikn'].nunique()
    npatients_by_category = data.groupby(category).apply(get_num_patients)
    most_common = npatients_by_category.sort_values(ascending=False)[0:top]
    return most_common

class DataSplitSummary:
    def __init__(self, X, split):
        self.patient_indices = X['ikn'].drop_duplicates(keep='last').index
        p = 'patients'
        s = 'sessions'
        self.total = {p: len(self.patient_indices),
                      s: len(X)}
        self.col = {p: (split, f'Patients (N={self.total[p]})'),
                    s: (split, f'Treatment Sessions (N={self.total[s]})')}
        self.X = X
        
    def num_sessions_per_patient_summary(self, summary_df):
        num_sessions = self.X.groupby('ikn').apply(len)
        mean = np.round(num_sessions.mean(), 1)
        std = np.round(num_sessions.std(), 1)
        summary_df.loc[('Number of Sessions', ''), self.col['patients']] = f'{mean}, SD ({std})'

    def avg_age_summary(self, summary_df):
        for with_respect_to, col in self.col.items():
            if with_respect_to == 'patients':
                age = self.X.loc[self.patient_indices, 'age']
            elif with_respect_to == 'sessions':
                age = self.X['age']
            q25, q75 = np.percentile(age, [25, 75])
            summary_df.loc[('Mean Age', ''), col] = f"{np.round(age.mean(),1)}, IQR [{q25}-{q75}]"

    def sex_summary(self, summary_df):
        for with_respect_to, col in self.col.items():
            total = self.total[with_respect_to]
            if with_respect_to == 'patients':
                F, M = self.X.loc[self.patient_indices, 'sex'].value_counts()
            elif with_respect_to == 'sessions':
                F, M = self.X['sex'].value_counts()
            summary_df.loc[('Sex', 'Female'), col] = f"{F} ({np.round(F/total*100, 1)}%)"
            summary_df.loc[('Sex', 'Male'), col] = f"{M} ({np.round(M/total*100, 1)}%)"
            
    def immigration_summary(self, summary_df):
        for with_respect_to, col in self.col.items():
            total = self.total[with_respect_to]
            if with_respect_to == 'patients':
                num_non_immigrants, num_immigrants = self.X.loc[self.patient_indices, 'is_immigrant'].value_counts()
                num_eng_speakers, num_non_eng_speakers = self.X.loc[self.patient_indices, 'speaks_english'].value_counts()
            elif with_respect_to == 'sessions':
                num_nonimmigrants, num_immigrants = self.X['is_immigrant'].value_counts()
                num_eng_speakers, num_non_eng_speakers = self.X['speaks_english'].value_counts()
            display = lambda x: f"{x} ({np.round(x/total*100, 1)}%)"
            summary_df.loc[('Immigration', 'Immigrant'), col] = display(num_immigrants)
            summary_df.loc[('Immigration', 'Non-immigrant'), col] = display(num_non_immigrants)
            summary_df.loc[('Immigration', 'English Speaker'), col] = display(num_eng_speakers)
            summary_df.loc[('Immigration', 'Non-English Speaker'), col] = display(num_non_eng_speakers)
            
    def other_summary(self, summary_df, col, display_name, top_items):
        """summarize items not in the top_items
        """
        other = self.X[~self.X[col].isin(top_items.keys())]
        num_patients = other['ikn'].nunique()
        num_sessions = len(other)
        for num, wrt in [(num_patients, 'patients'), (num_sessions, 'sessions')]:
            summary_df.loc[(display_name, 'Other'), self.col[wrt]] = f"{num} ({np.round(num/self.total[wrt]*100, 1)}%)"

    def regimen_summary(self, summary_df, col='regimen', display_name='Regimen', top=5):
        top_regimens = most_common_by_category(self.X, category=col, top=top)
        for regimen, num_patients in top_regimens.iteritems():
            summary_df.loc[(display_name, regimen), self.col['patients']] = \
                f"{num_patients} ({np.round(num_patients/self.total['patients']*100, 1)}%)"
            num_sessions = sum(self.X[col] == regimen)
            summary_df.loc[(display_name, regimen), self.col['sessions']] = \
                f"{num_sessions} ({np.round(num_sessions/self.total['sessions']*100, 1)}%)"
        # summarize the regimens not in top_regimens
        self.other_summary(summary_df, col, display_name, top_regimens)

    def cancer_location_summary(self, summary_df, col='curr_topog_cd', display_name='Cancer Location', top=5):
        top_cancer_locations = most_common_by_category(self.X, category=col, top=top)
        for cancer_location_code, num_patients in top_cancer_locations.iteritems():
            cancer_location = cancer_code_mapping[cancer_location_code]
            row = (display_name, cancer_location)
            summary_df.loc[row, self.col['patients']] = f"{num_patients} ({np.round(num_patients/self.total['patients']*100, 1)}%)"
            num_sessions = sum(self.X[col] == cancer_location_code)
            summary_df.loc[row, self.col['sessions']] = f"{num_sessions} ({np.round(num_sessions/self.total['sessions']*100, 1)}%)"
        # summarize the cancer locations not in top_cancer_locations
        self.other_summary(summary_df, col, display_name, top_cancer_locations)

    def cancer_type_summary(self, summary_df, col='curr_morph_cd', display_name='Cancer Type', top=5):
        top_cancer_types = most_common_by_category(self.X, category='curr_morph_cd', top=top)
        for cancer_type_code, num_patients in top_cancer_types.iteritems():
            cancer_type = cancer_code_mapping[cancer_type_code]
            row = (display_name, cancer_type)
            summary_df.loc[row, self.col['patients']] = f"{num_patients} ({np.round(num_patients/self.total['patients']*100, 1)}%)"
            num_sessions = sum(self.X[col] == cancer_type_code)
            summary_df.loc[row, self.col['sessions']] = f"{num_sessions} ({np.round(num_sessions/self.total['sessions']*100, 1)}%)"
        # summarize the cancer types not in top_cancer_types
        self.other_summary(summary_df, col, display_name, top_cancer_types)
        
    def combordity_summary(self, summary_df):
        for with_respect_to, col in self.col.items():
            total = self.total[with_respect_to]
            if with_respect_to == 'patients':
                num_non_diabetes, num_diabetes = self.X.loc[self.patient_indices, 'diabetes'].value_counts()
                num_non_ht, num_ht = self.X.loc[self.patient_indices, 'hypertension'].value_counts()
            elif with_respect_to == 'sessions':
                num_non_diabetes, num_diabetes = self.X['diabetes'].value_counts()
                num_non_ht, num_ht = self.X['hypertension'].value_counts()
            display = lambda x: f"{x} ({np.round(x/total*100, 1)}%)"
            summary_df.loc[('Combordity', 'Diabetes'), col] = display(num_diabetes)
            summary_df.loc[('Combordity', 'Non-diabetes'), col] = display(num_non_diabetes)
            summary_df.loc[('Combordity', 'Hypertension'), col] = display(num_ht)
            summary_df.loc[('Combordity', 'Non-hypertension'), col] = display(num_non_ht)
    
    def get_summary(self, summary_df, include_combordity=False):
        self.num_sessions_per_patient_summary(summary_df)
        self.avg_age_summary(summary_df)
        self.immigration_summary(summary_df)
        self.sex_summary(summary_df)
        self.regimen_summary(summary_df, top=2)
        self.cancer_location_summary(summary_df, top=3)
        self.cancer_type_summary(summary_df, top=3)
        if include_combordity:
            self.combordity_summary(summary_df)

def data_splits_summary(eval_models, save_dir, include_combordity=False):
    model_data = eval_models.orig_data
    summary_df = pd.DataFrame(index=twolevel, columns=twolevel)
    # All population summary
    dss = DataSplitSummary(model_data, 'All')
    dss.get_summary(summary_df, include_combordity=include_combordity)
    # Train, Valid, Test split population summary
    for split, Y in eval_models.labels.items():
        X = model_data.loc[Y.index]
        dss = DataSplitSummary(X, split)
        dss.get_summary(summary_df, include_combordity=include_combordity)
        
    summary_df.to_csv(f'{save_dir}/data_splits_summary.csv')
    return summary_df

def feature_summary(eval_models, prep, target_keyword, save_dir):
    df = prep.dummify_data(eval_models.orig_data)
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
    summary.to_csv(f'{save_dir}/feature_summary.csv')
    return summary

def nadir_summary(df, output_path, cytopenia='Neutropenia'):
    if cytopenia not in {'Neutropenia', 'Anemia', 'Thrombocytopenia'}: 
        raise ValueError('cytopenia must be one of Neutropneia, Anemia, or Thrombocytopenia')
    cycle_lengths = dict(df[['regimen', 'cycle_length']].values)
    result = {}
    for regimen, group in df.groupby('regimen'):
        cycle_length = int(cycle_lengths[regimen])
        group = group[range(0, cycle_length)]
        result[regimen] = {'NSessions': len(group), 'Cycle Length': cycle_length}
        for grade, thresholds in cytopenia_gradings.items():
            if cytopenia not in thresholds: continue
            thresh = thresholds[cytopenia]
            # get cytopenia rate for all sessions (regardless of day cytopenia occured)
            cytopenia_mask = group.min(axis=1).dropna() < thresh
            cytopenia_rate, cytopenia_std = cytopenia_mask.mean(), cytopenia_mask.std()
            margin_of_error = 1.96*cytopenia_std/np.sqrt(len(cytopenia_mask)) # z value = 1.96 for 95% confidence interval
            ci_lower, ci_upper = cytopenia_rate - margin_of_error, cytopenia_rate + margin_of_error
            cytopenia_rate, ci_lower, ci_upper = cytopenia_rate.round(2), max(0, ci_lower.round(2)), ci_upper.round(2)
            if grade == 'Grade 2':
                rate_per_day = [(group[day].dropna() < thresh).mean() for day in range(0,cycle_length)] # get the cytopenia rate for each day
                nadir_day = np.argmax(rate_per_day)
                result[regimen]['Nadir Day'] = nadir_day+1 # set day 1 as day of administration (not day 0)
            result[regimen][f'{grade} {cytopenia} Rate (<{thresh})'] = f'{cytopenia_rate} ({ci_lower}-{ci_upper})'
    summary_df = pd.DataFrame(result).T
    summary_df.to_csv(f'{output_path}/data/analysis/nadir_{cytopenia}_summary.csv')
    return summary_df
        
# Post Training Analysis
def group_pred_by_outcome(df, event='ACU'):
    """
    Group the predictions by outcome, collapse multiple chemo treatments into one outcome

    E.g. if chemo happens on Jan 1 and Jan 14, ED visit happens on Jan 20, and our lookahead window is 30 days, 
         then the outcome for predicting ED visit within 30 days would be a true positive if either Jan 1 and Jan 14 
         trigger a warning and a false negative if neither do

    Currrently only supports ED/H event outcomes
    """
    result = {} # index: prediction
    for ikn, ikn_group in df.groupby('ikn'):
        events = ['ED', 'H'] if event == 'ACU' else [event]
        for event in events:
            for date, date_group in ikn_group.groupby(f'next_{event}_date'):
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
    return -1

class SubgroupPerformanceSummary:
    def __init__(self, algorithm, eval_models, target_types,
                 pred_thresh=0.2, split='Test', display_ci=False, load_ci=False):
        self.model_dir = eval_models.output_path
        self.algorithm = algorithm
        self.split = split
        self.Y = eval_models.labels[split]
        self.data = eval_models.orig_data.loc[self.Y.index]
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
        for cat_feature, num_patients in most_common_by_category(self.data, category=category, top=3).iteritems():
            mask = self.data[category] == cat_feature
            Y_subgroup = self.Y[mask] 
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
        filepath = f'{output_path}/best_params/{algorithm}_classifier_best_param.pkl'
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
