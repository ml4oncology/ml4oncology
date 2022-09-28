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
import itertools
import pandas as pd
import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from functools import partial
from scipy.stats import pearsonr
from sklearn.metrics import (average_precision_score, precision_score, recall_score, roc_auc_score)

from src.config import (root_path, regiments_folder, all_observations,
                        intent_mapping, symptom_cols, clean_variable_mapping, 
                        eGFR_params, nn_solvers, nn_activations)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', datefmt='%I:%M:%S')

twolevel = pd.MultiIndex.from_product([[], []])

###############################################################################
# I/O
###############################################################################
def initialize_folders(output_path, extra_folders=None):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        if extra_folders is None: extra_folders = []
        main_folders = ['confidence_interval', 'perm_importance', 'best_params', 'predictions', 'tables', 'figures']
        figure_folders = ['important_features', 'curves', 'subgroup_performance', 'rnn_train_performance', 'decision_curve']
        figure_folders = [f'figures/{folder}' for folder in figure_folders]
        for folder in main_folders + figure_folders + extra_folders:
            os.makedirs(f'{output_path}/{folder}')

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

def load_chemo_df(main_dir, includes_next_visit=True):
    dtype = {'ikn': str, 'lhin_cd': str, 'curr_morph_cd': str}
    chemo_df = pd.read_csv(f'{main_dir}/data/chemo_processed.csv', dtype=dtype)
    chemo_df['visit_date'] = pd.to_datetime(chemo_df['visit_date'])
    if includes_next_visit:
        chemo_df['next_visit_date'] = pd.to_datetime(chemo_df['next_visit_date'])
    return chemo_df

def load_reviewed_regimens():
    """
    Get the annotated regimens with their cycle lengths, name relabelings, splits, etc
    """
    df = pd.read_csv(f'{root_path}/{regiments_folder}/regimens.csv', dtype=str)
    df = clean_string(df, df.columns)

    # convert select columns to floats
    df['shortest_interval'] = df['shortest_interval'].replace('unclear', -1)
    float_cols = df.columns.drop(['regimen', 'relabel', 'reason', 'notes'])
    df[float_cols] = df[float_cols].astype(float)
    
    # ensure regimen names are all lowercase
    df['regimen'] = df['regimen'].str.lower()
    df['relabel'] = df['relabel'].str.lower()
    
    df = df.rename(columns={'split_number_first_dose_second_c': 'split_at_this_cycle',
                            'split_second_component_shortest_': 'cycle_length_after_split',
                            'shortest_interval': 'cycle_length'})
    df['shortest_cycle_length'] = df[['cycle_length', 'cycle_length_after_split']].min(axis=1)
    return df

def load_included_regimens(criteria=None):
    """
    Args:
        criteria (str or None): inclusion criteria for regimens based on different projects, 
                                either None (no critera), 'cytotoxic' for CYTOPENIA, or 'cisplatin_containing' for CAN
    """
    if criteria not in {None, 'cytotoxic', 'cisplatin_containing'}:
        raise ValueError('criteria must be either None, "cytotoxic", or "cisplatin_containing"')
    df = load_reviewed_regimens()

    # filter outpatient regimens, hematological regimens, non-IV administered regimens for all projects
    # NOTE: all regimens that are relabeled are kept
    mask = df['iv_non_hematological_outpatient'] == 0
    df = df[~mask]
    
    if criteria is not None:
        # keep only selected reigmens for the project or regimens that are relabeled
        df = df[(df[criteria] == 1) | df['relabel'].notnull()]
        
        if criteria == 'cytotoxic':
            # remove regimens with unclear cycle lengths
            df = df[df['shortest_cycle_length'] != -1]
    
    # filter out regimens relabeled to an excluded regimen
    df = df[df['relabel'].isin(df['regimen']) | df['relabel'].isnull()]

    return df

###############################################################################
# Multiprocessing
###############################################################################
def parallelize(generator, worker, processes=16):
    pool = mp.Pool(processes=processes)
    result = pool.map(worker, generator)
    pool.close()
    pool.join() # wait for all threads
    result = list(itertools.chain(*result))
    return result

def split_and_parallelize(data, worker, split_by_ikns=True, processes=16):
    """
    Split up the data and parallelize processing of data
    
    Args:
        data: array-like, DataFrame or tuple of DataFrames sharing the same patient ids
        split_by_ikns: split up the data by patient ids
    """
    generator = []
    if split_by_ikns:
        ikns = data[0]['ikn'] if isinstance(data, tuple) else data['ikn']
        ikn_groupings = np.array_split(ikns.unique(), processes)
        if isinstance(data, tuple):
            for ikn_grouping in ikn_groupings:
                items = tuple(df[df['ikn'].isin(ikn_grouping)] for df in data)
                generator.append(items)
        else:
            for ikn_grouping in ikn_groupings:
                item = data[ikns.isin(ikn_grouping)]
                generator.append(item)
    else:
        # splits df into x number of partitions, where x is number of processes
        generator = np.array_split(data, processes)
    return parallelize(generator, worker, processes=processes)

###############################################################################
# Misc (Cleaners, Forward Fillers, Groupers)
###############################################################################
def clean_string(df, cols):
    # remove first two characters "b'" and last character "'"
    for col in cols:
        mask = df[col].str.startswith("b'") & df[col].str.endswith("'")
        df.loc[mask, col] = df.loc[mask, col].str[2:-1].values
    return df

def replace_rare_col_entries(df, cols, with_respect_to='patients', n=6, verbose=False):
    for col in cols:
        if with_respect_to == 'patients':
            # replace unique values with less than n patients to 'Other'
            counts = df.groupby(col).apply(lambda group: len(set(group['ikn'])))
        elif with_respect_to == 'rows':
            # replace unique values that appears less than n number of rows in the dataset to 'Other'
            counts = df[col].value_counts()
        replace_values = counts.index[counts < n]
        if verbose: 
            logging.info(f'The following {col} entries have less than {n} {with_respect_to} '
                         f'and will be replaced with "Other": {replace_values.tolist()}')
        df.loc[df[col].isin(replace_values), col] = 'Other'
    return df

def pandas_ffill(df):
    # uh...unfortunately pandas ffill is super slow when scaled up
    return df.ffill(axis=1)[0].values

def numpy_ffill(df):
    # courtesy of stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    arr = df.values.astype(float)
    num_rows, num_cols = arr.shape
    mask = np.isnan(arr)
    indices = np.where(~mask, np.arange(num_cols), 0)
    np.maximum.accumulate(indices, axis=1, out=indices)
    arr[mask] = arr[np.nonzero(mask)[0], indices[mask]]
    return arr[:, -1]

def group_observations(observations, freq_map):
    grouped_observations = {}
    for obs_code, obs_name in observations.items():
        if obs_name in grouped_observations:
            grouped_observations[obs_name].append(obs_code)
        else:
            grouped_observations[obs_name] = [obs_code]

    for obs_name, obs_codes in grouped_observations.items():
        # sort observation codes based on their number of occurences / observation frequencies
        grouped_observations[obs_name] = freq_map[obs_codes].sort_values(ascending=False).index.tolist()
   
    return grouped_observations

###############################################################################
# Bootstrap Confidence Interval 
###############################################################################
def bootstrap_sample(data, random_seed):
    np.random.seed(random_seed)
    N = len(data)
    weights = np.random.random(N) 
    return data.sample(n=N, replace=True, random_state=random_seed, weights=weights)
    
def bootstrap_worker(bootstrap_partition, Y_true, Y_pred):
    """
    Compute bootstrapped AUROC/AUPRC scores by resampling the labels and predictions together
    and recomputing the AUROC/AUPRC
    """
    scores = []
    for random_seed in bootstrap_partition:
        y_true = bootstrap_sample(Y_true, random_seed)
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
        logging.warning(f'{n_skipped} bootstraps with no pos examples were skipped, '
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

###############################################################################
# Data Descriptions
###############################################################################
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
    rename_variable_mapping = {'is_immigrant': 'immigrated_to_canada',
                               'speaks_english': 'immigrated,_speaks_english',
                               'sex': 'female_sex',
                               'immediate': 'first_dose_of'}
    rename_variable_mapping.update({name: f'esas_{name}_score' for name in symptom_cols if 'grade' not in name})
            
    def clean_name(name):
        # swap cause and event
        for event in ['ED', 'H']:
            for cause in ['INFX', 'TR', 'GI']:
                name = name.replace(f'{cause}_{event}', f'{event}_{cause}')
        for mapping in [clean_variable_mapping, rename_variable_mapping]:
            for orig, new in mapping.items():
                name = name.replace(orig, new)
        name = name.replace('_', ' ')
        name = name.title()
        for substr in ['Ed', 'Icd', 'Other', 'Esas']:
            name = name.replace(substr, substr.upper()) # capitalize certain substrings
        for substr in [' Of ', ' To ']:
            name = name.replace(substr, substr.lower()) # lowercase certain substrings
        if name.startswith('Intent ') and name[-1] in intent_mapping:
            name = f'{name[:-1]}{intent_mapping[name[-1]].title()}' # get full intent description
        elif name.startswith('Regimen '):
            name = f"Regimen {name.split(' ')[-1].upper()}" # capitalize all regimen names
        elif name.endswith(')'):
            name, unit = name.split('(')
            name = '('.join([name, unit.lower()]) # lowercase the units
            name = name.replace('/l)', '/L)') # recaptialize the Liter lol
            
        return name
            
    return [clean_name(name) for name in names]

def get_observation_units():
    filename = f'{root_path}/data/olis_units.pkl'
    with open(filename, 'rb') as file:
        units_map = pickle.load(file)
    
    # group the observation units together with same observation name
    grouped_units_map = {}
    for obs_code, obs_name in all_observations.items():
        obs_name = f'baseline_{obs_name}_count'
        if obs_name in grouped_units_map:
            grouped_units_map[obs_name].append(units_map[obs_code])
        else:
            grouped_units_map[obs_name] = [units_map[obs_code]]

    # just keep the first observation unit, since they are all the same
    for obs_name, units in grouped_units_map.items():
        assert len(set(units)) == 1
        grouped_units_map[obs_name] = units[0]
        
    return grouped_units_map

def get_units():
    units_map = get_observation_units()
    units_map.update({'age': 'years',
                      'body_surface_area': 'm2', 
                      'cisplatin_dosage': 'mg/m2',
                      'baseline_eGFR': 'ml/min/1.73m2'})
    units_map.update({var: 'yes/no' for var in ['immediate_new_regimen', 'hypertension', 'diabetes', 
                                                'speaks_english', 'is_immigrant', 'sex']})
    return units_map

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

###############################################################################
# Predictions
###############################################################################
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
    """
    Find the closest prediction threshold that will achieve the desired metric score
    using binary search
    """
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

###############################################################################
# Time Intervals
###############################################################################
def time_to_target_after_alarm(eval_models, event_dates, target_event, target_date_col, 
                               split='Test', algorithm='ENS', pred_thresh=0.5):
    """
    Get the time to target after the first alarm incident (risk prediction > threshold) for each patient
    """
    df = pd.DataFrame()
    df[[target_date_col, 'visit_date']] = event_dates[[target_date_col, 'visit_date']]
    df['pred'] = eval_models.preds[split][algorithm][target_event]
    df['ikn'] = eval_models.orig_data['ikn']
    df['label'] = eval_models.labels[split][target_event]
    
    # get first incident when risk prediction surpassed the threshold for each patient
    # NOTE: data is already sorted by visit date, so first alarm incident is always the first row of the group
    mask = df['pred'] > pred_thresh
    first_alarm = df[mask].groupby('ikn').first()
    # get the time to target
    time_to_target = (first_alarm[target_date_col] - first_alarm['visit_date']).dt.days
    logging.info(f'Alarm = high risk of {target_event} (risk > {pred_thresh})')
    logging.info(f'{len(first_alarm)} patients had alarms. '
                 f'Among them, {sum(time_to_target.isnull())} patients did not experience the event at all / were censored.')

    # get patients whose risk prediction never surpassed the threshold
    low_risk_ikns = set(df['ikn']).difference(first_alarm.index)
    # determine whether they've experienced the target
    mask = df['ikn'].isin(low_risk_ikns)
    experienced_target = df[mask].groupby('ikn')['label'].any()
    logging.info(f'{len(low_risk_ikns)} patients had no alarms. '
                 f'Among them, {sum(experienced_target)} patients did experience {target_event}.')
    
    return time_to_target

def time_to_alarm_after_service(eval_models, event_dates, target_event,
                                split='Test', algorithm='ENS', pred_thresh=0.5):
    """
    Get the time (in months) to first alarm incident after the palliative consultation service for each patient
    """
    df = pd.DataFrame()
    df['preds'] = eval_models.preds[split][algorithm][target_event]
    df['ikn'] = eval_models.orig_data['ikn']
    df['service_date'] = event_dates['palliative_consultation_service_date']
    df['visit_date'] = event_dates['visit_date']

    # remove sessions where no early pallative consultation service was requested
    mask = df['service_date'].notnull()
    logging.info('Filtering out sessions where palliative consultation service was not requested prior to those sessions.\n'
                 f'{sum(mask)} sessions remain out of {len(mask)} total sessions.\n'
                 f'{df.loc[mask, "ikn"].nunique()} patients remain out of {df["ikn"].nunique()} total patients.\n')
    df = df[mask]

    # remove sessions where alarm was not triggered
    mask = df['preds'] > pred_thresh
    logging.info(f'Filtering out sessions where risk of target event did not exceed {pred_thresh}.\n'
                 f'{sum(mask)} sessions remain.\n'
                 f'{df.loc[mask, "ikn"].nunique()} patients remain.\n')
    df = df[mask]

    # take the first alarm incident of each patient
    df = df.groupby('ikn').first()

    # get the time to first alarm after palliative consultation date
    diff_month = lambda d1, d2: (d1.dt.year - d2.dt.year)*12 + d1.dt.month - d2.dt.month
    time_to_alarm = diff_month(df['visit_date'], df['service_date'])
    time_to_alarm = time_to_alarm.clip(upper=time_to_alarm.quantile(q=0.999))
    time_to_alarm = time_to_alarm.sort_values()
    
    return time_to_alarm

###############################################################################
# Estimated Glomerular Filteration Rate
###############################################################################
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

###############################################################################
# Hyperparameters
###############################################################################
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

###############################################################################
# Pearson Correlation
###############################################################################
def get_pearson_matrix(df, target_keyword, output_path):
    dtypes = df.dtypes
    cols = dtypes[~(dtypes == object)].index
    cols = cols.drop('ikn')
    target_cols = cols[cols.str.contains(target_keyword)]
    feature_cols = cols[~cols.str.contains(target_keyword)]
    
    pearson_matrix = pd.DataFrame(columns=feature_cols, index=target_cols)
    for target in target_cols:
        for feature in tqdm(feature_cols):
            data = df[~df[feature].isnull()]
            corr, prob = pearsonr(data[target], data[feature])
            pearson_matrix.loc[target, feature] = np.round(corr, 3)
    pearson_matrix = pearson_matrix.T
    pearson_matrix.index = get_clean_variable_names(pearson_matrix.index)
    pearson_matrix.columns = pearson_matrix.columns.str.replace(target_keyword, '')
    pearson_matrix.to_csv(f'{output_path}/tables/pearson_matrix.csv', index_label='index')
    return pearson_matrix
