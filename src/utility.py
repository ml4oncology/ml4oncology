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
from collections import defaultdict
from functools import partial
import itertools
import os
import pickle

from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import (
    average_precision_score, 
    precision_score, 
    recall_score, 
    roc_auc_score
)
import multiprocessing as mp
import numpy as np
import pandas as pd

from src.config import (
    max_chemo_date,
    all_observations,
    clean_variable_mapping, 
    eGFR_params, 
    intent_mapping, 
    nn_solvers, nn_activations,
    regiments_folder, root_path, 
    symptom_cols, 
)

import logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', 
    datefmt='%I:%M:%S'
)

twolevel = pd.MultiIndex.from_product([[], []])

###############################################################################
# I/O
###############################################################################
def initialize_folders(output_path, extra_folders=None):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        if extra_folders is None: extra_folders = []
        main_folders = [
            'confidence_interval', 'perm_importance', 'best_params', 
            'predictions', 'tables', 'figures'
        ]
        figure_folders = [
            'important_features', 'curves', 'subgroup_performance', 
            'rnn_train_performance', 'decision_curve'
        ]
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
        col = 'next_visit_date'
        chemo_df[col] = pd.to_datetime(chemo_df[col])
    return chemo_df

def load_reviewed_regimens():
    """Get the annotated regimens with their cycle lengths, name relabelings, 
    splits, etc
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
    
    # relabel names
    name_map = {
        'split_number_first_dose_second_c': 'split_at_this_cycle',
        'split_second_component_shortest_': 'cycle_length_after_split',
        'shortest_interval': 'cycle_length'
    }
    df = df.rename(columns=name_map)
    
    # get shortest cycle length
    cols = ['cycle_length', 'cycle_length_after_split']
    df['shortest_cycle_length'] = df[cols].min(axis=1)
    
    return df

def load_included_regimens(criteria=None):
    """
    Args:
        criteria (str): inclusion criteria for regimens based on different 
            projects. If None, all regimens are included. Options include:
            1. 'cytotoxic' for CYTOPENIA
            2. 'cisplatin_containing' for CAN
    """
    if criteria not in {None, 'cytotoxic', 'cisplatin_containing'}:
        raise ValueError('criteria must be either None, "cytotoxic", or '
                         '"cisplatin_containing"')
    df = load_reviewed_regimens()

    # filter outpatient regimens, hematological regimens, non-IV administered 
    # regimens for all projects. NOTE: all regimens that are relabeled are kept
    mask = df['iv_non_hematological_outpatient'] == 0
    df = df[~mask]
    
    if criteria is not None:
        # keep only selected reigmens for the project or regimens that are 
        # relabeled
        df = df[(df[criteria] == 1) | df['relabel'].notnull()]
        
        if criteria == 'cytotoxic':
            # remove regimens with unclear cycle lengths
            df = df[df['shortest_cycle_length'] != -1]
    
    # filter out regimens relabeled to an excluded regimen
    df = df[df['relabel'].isin(df['regimen']) | df['relabel'].isnull()]

    return df

def get_mapping_from_textfile(filepath, replacement=None):
    """Extracts the mapping between name and code into a 
    dictionary
    
    Contents of the textfile should be of the form
    Name1 = code1
    Name1 = code2
    Name2 = code3
    
    Args:
        replacement (dict): A mapping of original name (str) to desired 
            name (str)
    """
    if replacement is None: 
        replacement = {}
        
    with open(filepath, 'r') as f:
        lines = f.readlines()

    mapping = defaultdict(list)
    for line in lines:
        code, name = line.strip().split(' = ')
        name = replacement.get(name, name)
        name = name.split(',')[0] 
        mapping[name].append(code)
        
    return mapping

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
    """Split up the data and parallelize processing of data
    
    Args:
        data: Supports a sequence, pd.DataFrame, or tuple of pd.DataFrames 
            sharing the same patient ids
        split_by_ikns (bool): If True, split up the data by patient ids
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
def replace_rare_categories(
    df, 
    catcols, 
    with_respect_to='patients', 
    n=6, 
    verbose=False
):
    for col in catcols:
        if with_respect_to == 'patients':
            # replace unique categories with less than n patients to 'Other'
            counts = df.groupby(col).apply(lambda g: g['ikn'].nunique())
        elif with_respect_to == 'rows':
            # replace unique categories that appears less than n number of rows
            # in the dataset to 'Other'
            counts = df[col].value_counts()
        replace_cats = counts.index[counts < n]
        if verbose:
            logging.info(f'The following {col} categories have less than {n} '
                         f'{with_respect_to} and will be replaced with "Other"'
                         f': {replace_cats.tolist()}')
        df.loc[df[col].isin(replace_cats), col] = 'Other'
    return df

def group_observations(observations, freq_map):
    grouped_observations = {}
    for obs_code, obs_name in observations.items():
        if obs_name in grouped_observations:
            grouped_observations[obs_name].append(obs_code)
        else:
            grouped_observations[obs_name] = [obs_code]

    for obs_name, obs_codes in grouped_observations.items():
        # sort observation codes based on their prevalence
        obs_codes = freq_map[obs_codes].sort_values(ascending=False).index
        grouped_observations[obs_name] = obs_codes.tolist()
   
    return grouped_observations
    
def pandas_ffill(df):
    # uh...unfortunately pandas ffill is super slow when scaled up
    return df.ffill(axis=1)[0].values

def numpy_ffill(df):
    # Ref: stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    arr = df.values.astype(float)
    num_rows, num_cols = arr.shape
    mask = np.isnan(arr)
    indices = np.where(~mask, np.arange(num_cols), 0)
    np.maximum.accumulate(indices, axis=1, out=indices)
    arr[mask] = arr[np.nonzero(mask)[0], indices[mask]]
    return arr[:, -1]
    
def clean_string(df, cols):
    # remove first two characters "b'" and last character "'"
    for col in cols:
        mask = df[col].str.startswith("b'") & df[col].str.endswith("'")
        df.loc[mask, col] = df.loc[mask, col].str[2:-1].values
    return df

def get_years_diff(df, col1, col2):
    return df[col1].dt.year - df[col2].dt.year

def get_first_alarms_or_last_treatment(df, pred_thresh, verbose=False):
    """Get each patient's first alarm incident (the session where the first
    alarm occured). If no alarms were ever triggered, get the patient's last 
    session
    """
    df['predicted'] = df['predicted_prob'] > pred_thresh
    first_alarm = df[df['predicted']].groupby('ikn').first()
    low_risk_ikns = set(df['ikn']).difference(first_alarm.index)
    last_session = df[df['ikn'].isin(low_risk_ikns)].groupby('ikn').last()
    if verbose:
        logging.info('Number of patients with first alarm incidents '
                     f'(risk > {pred_thresh:.2f}): {len(first_alarm)}. Number '
                     'of patients with no alarm incidents (take the last '
                     f'session): {len(last_session)}')
    df = pd.concat([first_alarm, last_session])
    return df

###############################################################################
# Bootstrap Confidence Interval 
###############################################################################
def bootstrap_sample(data, random_seed):
    np.random.seed(random_seed)
    N = len(data)
    weights = np.random.random(N) 
    return data.sample(
        n=N, replace=True, random_state=random_seed, weights=weights
    )
    
def bootstrap_worker(bootstrap_partition, Y_true, Y_pred):
    """Compute bootstrapped AUROC/AUPRC scores by resampling the labels and 
    predictions together and recomputing the AUROC/AUPRC
    """
    scores = []
    for random_seed in bootstrap_partition:
        y_true = bootstrap_sample(Y_true, random_seed)
        y_pred = Y_pred[y_true.index]
        if y_true.nunique() < 2:
            continue
        scores.append((
            roc_auc_score(y_true, y_pred), 
            average_precision_score(y_true, y_pred)
        ))
    return scores

def compute_bootstrap_scores(Y_true, Y_pred, n_bootstraps=10000, processes=32):
    worker = partial(bootstrap_worker, Y_true=Y_true, Y_pred=Y_pred)
    scores = split_and_parallelize(
        range(n_bootstraps), worker, split_by_ikns=False, processes=processes
    )
    
    n_skipped = n_bootstraps - len(scores)
    if n_skipped > 0: 
        logging.warning(f'{n_skipped} bootstraps with no pos examples were '
                        'skipped, skipped bootstraps will be replaced with '
                        'original score')
        # fill skipped boostraps with the original score
        orig_score = [(
            roc_auc_score(Y_true, Y_pred), 
            average_precision_score(Y_true, Y_pred),
        )]
        scores += orig_score * n_skipped
        
    return scores

def nadir_bootstrap_worker(bootstrap_partition, df, days, thresh):
    """Compute bootstrapped nadir days by resampling patients with replacement 
    and recomputing nadir day. Used for computing confidence interval of actual
    nadir day
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
def get_nunique_categories(df):
    catcols = df.dtypes[df.dtypes == object].index.tolist()
    return pd.DataFrame(
        df[catcols].nunique(), columns=['Number of Unique Categories']
    ).T

def get_nmissing(df, verbose=False):
    missing = df.isnull().sum() # number of nans for each column
    missing = missing[missing != 0] # remove columns without missing values
    missing = pd.DataFrame(missing, columns=['Missing (N)'])
    missing['Missing (%)'] = (missing['Missing (N)'] / len(df) * 100).round(3)
        
    if verbose:
        other = [
            'intent_of_systemic_treatment', 'lhin_cd', 'curr_morph_cd', 
            'curr_topog_cd', 'body_surface_area'
        ]
        idx = missing.index
        mapping = {
            'lab tests': missing.loc[idx.str.contains('baseline')],
            'symptom values': missing.loc[idx[idx.isin(symptom_cols)]],
            'other data': missing.loc[idx[idx.isin(other)]]
        }
        for name, miss in mapping.items():
            miss_percentage = miss['Missing (%)']
            logging.info(f'{miss_percentage.min()}%-{miss_percentage.max()}% of '
                         f'{name} were missing before treatment sessions')
        
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
        names: A sequence of variable names (str)
    """
    return [get_clean_variable_name(name) for name in names]

def get_clean_variable_name(name):
    """
    Args:
        name (str): variable name
    """
    # swap cause and event
    for event in ['ED', 'H']:
        for cause in ['INFX', 'TR', 'GI']:
            name = name.replace(f'{cause}_{event}', f'{event}_{cause}')
    
    # rename variables
    rename_variable_mapping = {
        'is_immigrant': 'immigrated_to_canada',
        'speaks_english': 'immigrated,_speaks_english',
        'sex': 'female_sex',
        'immediate': 'first_dose_of'
    }
    rename_variable_mapping.update(
        {name: f'esas_{name}_score' for name in symptom_cols 
         if 'grade' not in name}
    )
    for mapping in [clean_variable_mapping, rename_variable_mapping]:
        for orig, new in mapping.items():
            name = name.replace(orig, new)
            
    if name.startswith('world_region'):
        var, region = name.rsplit('_', 1)
        name = f"{var.replace('_', ' ').title()} {region}"
    else:
        name = name.replace('_', ' ').title()
    
    # capitalize certain substrings
    for substr in ['Ed', 'Icd', 'Other', 'Esas']:
        name = name.replace(substr, substr.upper())
    # lowercase certain substrings
    for substr in [' Of ', ' To ']:
        name = name.replace(substr, substr.lower())
     
    if name.startswith('Intent ') and name[-1] in intent_mapping:
        # get full intent description
        name = f'{name[:-1]}{intent_mapping[name[-1]].title()}'
    elif name.startswith('Regimen '):
        # capitalize all regimen names
        name = f"Regimen {name.split(' ')[-1].upper()}"
    elif name.endswith(')'):
        name, unit = name.split('(')
        name = '('.join([name, unit.lower()]) # lowercase the units
        name = name.replace('/l)', '/L)') # recaptialize the Liter lol

    return name

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
    units_map.update({
        'age': 'years',
        'body_surface_area': 'm2', 
        'cisplatin_dosage': 'mg/m2',
        'baseline_eGFR': 'ml/min/1.73m2'
    })
    bool_cols = [
        'immediate_new_regimen', 'hypertension', 'diabetes', 'speaks_english', 
        'is_immigrant', 'sex'
    ]
    units_map.update({var: 'yes/no' for var in bool_cols})
    return units_map

def get_cyto_rates(df, thresh):
    """Get cytopenia rate on nth day of chemo cycle, based on available 
    measurements
    """
    cytopenia_rates_per_day = []
    for day, measurements in df.iteritems():
        measurements = measurements.dropna()
        cytopenia_rate = (measurements < thresh).mean()
        cytopenia_rates_per_day.append(cytopenia_rate)
    cytopenia_rates_per_day = pd.Index(cytopenia_rates_per_day).fillna(0)
    return cytopenia_rates_per_day 
    
def most_common_categories(
    data, 
    catcol='regimen', 
    with_respect_to='sessions', 
    top=5
):
    # most common categories in a categorical column (e.g. regimen, 
    # cancer location, cancer type) with respect to patients or sessions
    if with_respect_to == 'patients':
        get_num_patients = lambda group: group['ikn'].nunique()
        category_count = data.groupby(catcol).apply(get_num_patients)
    elif with_respect_to == 'sessions':
        category_count = data.groupby(catcol).apply(len)
    else:
        raise ValueError('Must be with respect to patients or sessions')
    most_common = category_count.sort_values(ascending=False)[0:top]
    most_common = most_common.to_dict()
    return most_common

###############################################################################
# Predictions & Metrics 
###############################################################################
def outcome_recall_score(
    Y_true, 
    Y_pred_bool, 
    target_event, 
    event_dates=None, 
    lookback_window='30d'
):
    """Compute the outcome-level recall/sensitivity. The proportion of events 
    where at least one warning was issued prior to the event 
    
    Args:
        event_dates (pd.DataFrame): table of relevant event dates (e.g. D_date,
            next_ED_date, visit_date, etc) associated with each session. Also 
            includes ikn (patient id)
        lookback_window (str): number of days prior to the event in which a 
            warning is valid, in the format '{int}d'
    """
    event_col_map = {
        'ACU': ['next_H_date', 'next_ED_date'], 
        'H': ['next_H_date'], 
        'ED': ['next_ED_date'], 
        'D': ['D_date']
    }
    if event_dates is None: 
        raise ValueError('Please provide the event dates')
    event_dates = event_dates.loc[Y_true.index]
    event_dates['true'] = Y_true
    event_dates['pred'] = Y_pred_bool
    
    if target_event.endswith('Mortality'):
        # e.g. target_event = '14d Mortality'
        lookback_window = target_event.split(' ')[0] 
        event = 'D'
    else:
        # event will be either ACU, ED, or H
        # e.g. target_event = 'INFX_H'
        event = target_event.split('_')[-1]
        
    if event not in event_col_map:
        raise ValueError(f'Does not support {target_event}')
    event_cols = event_col_map[event]
    
    # exclude sessions without outcome dates
    mask = False
    for col in event_cols:
        mask |= event_dates[col].notnull()
    
    # group predictions by outcome
    worker = partial(
        group_pred_by_outcome, event_cols=event_cols, 
        lookback_window=lookback_window
    )
    grouped_preds = split_and_parallelize(
        event_dates[mask], worker, processes=8
    )
    grouped_preds = pd.DataFrame(grouped_preds, columns=['chemo_idx', 'pred'])
    grouped_preds = grouped_preds.set_index('chemo_idx')
    
    # select the rows of interest
    result = pd.concat([
        event_dates.loc[grouped_preds.index], 
        event_dates[~mask]
    ])
    
    # update the predictions
    result.loc[grouped_preds.index, 'pred'] = grouped_preds['pred']
    
    return recall_score(result['true'], result['pred'], zero_division=1)
    
def group_pred_by_outcome(df, event_cols, lookback_window='30d'):
    """Group the predictions by outcome, collapse multiple chemo treatments 
    into one outcome.

    E.g. if ED visit happens on Jan 20, chemo happens on Jan 1 and Jan 14, and
         our lookback window is 30 days, then the outcome for predicting ED 
         visit within 30 days would be true positive if either Jan 1 or Jan 14 
         trigger a warning, and false negative if neither do

    Currrently only supports ED/H event and Death outcomes
    """
    lookback_window = pd.Timedelta(lookback_window)
    result = defaultdict(bool) # {index: prediction}, default to False
    for ikn, ikn_group in df.groupby('ikn'):
        for event in event_cols:
            for date, date_group in ikn_group.groupby(event):
                # only keep samples in which event occured within lookback 
                # window
                mask = date_group['visit_date'] >= date - lookback_window
                idx = date_group.index[-1]
                result[idx] = result[idx] or any(date_group.loc[mask, 'pred'])
    return list(result.items())

def equal_rate_pred_thresh(
    eval_models, 
    event_dates, 
    split='Valid', 
    alg='ENS', 
    target_event='365d Mortality'
):
    """Find the prediction threshold at which the alarm rate roughly equals the
    intervention rate during usual care (ensuring the warning system would use
    same amount of resource as usual care)
    
    For target of 365d Mortality, intervention is receiving palliative care 
    consultation service (PCCS)
    """
    # Extract relevant data
    df = pd.DataFrame()
    df['predicted_prob'] = eval_models.preds[split][alg][target_event]
    df['ikn'] = eval_models.orig_data.loc[df.index, 'ikn']
    
    # Get intervention rate (by patient, not session)
    if target_event == '365d Mortality':
        tmp = event_dates.loc[df.index, ['first_PCCS_date', 'ikn']]
        tmp = tmp.drop_duplicates()
        intervention_rate = tmp['first_PCCS_date'].notnull().mean()
    else:
        raise NotImplementedError(f'Target {target_event} is not supported yet')

    def scorer(thresh, df):
        # Take the first session in which an alarm was triggered for each patient
        # or the last session if alarms were never triggered
        df = get_first_alarms_or_last_treatment(df, thresh, verbose=False)
        # Get the alarm rate
        alarm_rate = df['predicted'].mean()
        return alarm_rate
    
    return binary_search(
        scorer, intervention_rate, 'alarm rate', df, swap_adjustment=True,
    )
    
def pred_thresh_binary_search(
    Y_pred_prob, 
    Y_true, 
    desired_target, 
    metric='precision',
    **kwargs
):
    """Finds the closest prediction threshold that will achieve the desired 
    metric score using binary search
    """
    score_func = {
        'precision': precision_score, 
        'sensitivity': recall_score, 
        'warning_rate': lambda y_true, y_pred: y_pred.mean()
    }
    
    def scorer(thresh, Y_true, Y_pred_prob, **kwargs):
        Y_pred_bool = Y_pred_prob > thresh
        return score_func[metric](Y_true, Y_pred_bool, **kwargs)
    
    swap_adjustment = metric in {'sensitivity', 'warning_rate'}
    
    return binary_search(
        scorer, desired_target, metric, Y_true, Y_pred_prob, 
        swap_adjustment=swap_adjustment, **kwargs
    )

def binary_search(
    scorer,
    desired_target, 
    metric, 
    *args, 
    swap_adjustment=False,
    **kwargs
):
    less_than_desired_adjust = lambda low, mid, high: (mid + 0.0001, high)
    more_than_desired_adjust = lambda low, mid, high: (low, mid - 0.0001)
    if swap_adjustment:
        less_than_desired_adjust, more_than_desired_adjust = \
        more_than_desired_adjust, less_than_desired_adjust
        
    cur_target = 0
    low, high = 0, 1
    while low <= high:
        mid = (high + low)/2
        cur_target = scorer(mid, *args, **kwargs)
        if abs(cur_target - desired_target) <= 0.005:
            return mid
        elif cur_target < desired_target:
            low, high = less_than_desired_adjust(low, mid, high)
        elif cur_target > desired_target:
            low, high = more_than_desired_adjust(low, mid, high)
            
    logging.warning(f'Desired {metric} {desired_target:.2f} could not be '
                    f'achieved. Closest {metric} achieved was {cur_target:.2f}')
    return mid
    
###############################################################################
# Time Intervals
###############################################################################
def time_to_x_after_y(
    eval_models, 
    event_dates,   
    x=None,
    y=None,
    split='Test', 
    algorithm='ENS', 
    target_event='365d Mortality',
    pred_thresh=0.5,
    verbose=True,
    care_name=None,
    clip=False
):
    """Get the time (in months) to event x after event y for each patient
    
    x/y can be first alarm incident, death, first palliative care consultation
    service (PCCS), last observation, etc.
    
    Args:
        x (str): event, either 'first_alarm', 'death', 'first_pccs', 'last_obs'
        y (str): event, either 'first_alarm', 'death', 'first_pccs', 'last_obs'
    """
    if x == y: raise ValueError('x and y must be different')
    
    # Get relevant data
    df = pd.DataFrame()
    df['predicted_prob'] = eval_models.preds[split][algorithm][target_event]
    df['ikn'] = eval_models.orig_data.loc[df.index, 'ikn']
    df[event_dates.columns] = event_dates.loc[df.index]
    
    # Take the first session in which an alarm was triggered for each patient
    # or the last session if alarms were never triggered
    df = get_first_alarms_or_last_treatment(df, pred_thresh, verbose=verbose)
    
    # get the time to event x after event y
    cols = {
        'last_obs': 'last_seen_date',
        'death': 'D_date',
        'first_alarm': 'visit_date', 
        'first_pccs': 'first_PCCS_date'
    }
    
    if x == 'first_alarm' or y == 'first_alarm':
        # remove patients who never had an alarm
        mask = df['predicted']
        if verbose:
            logging.info('Removing patients who never had an alarm. '
                         f'{sum(mask)} patients remain out of {len(mask)} '
                         'total patients.')
        df = df[mask]
    
    if x == 'death' or y == 'death':
        col = cols['death']
        # remove patients who never died
        mask = df[col].notnull()
        if verbose:
            logging.info(f'Removing patients who never died. {sum(mask)} '
                         f'patients remain out of {len(mask)} total patients.')
        df = df[mask]
        
    if x == 'first_pccs' or y == 'first_pccs':
        col = cols['first_pccs']
        
        if care_name == 'Model-Guided Care':
            # update where warning system initiated PCCS earlier than usual care
            mask = df['predicted']
            dates = df.loc[mask, [col, 'visit_date']]
            df[col][mask] = dates.min(axis=1)
            
        # remove patients who never received PCCS or 
        # received PCCS after cohort end date
        mask1 = df[col].isnull()
        mask2 = df[col] > max_chemo_date
        if verbose:
            logging.info(f'Removing {sum(mask1)} patients that never received PCCS.')
            logging.info(f'Removing {sum(mask2)} patients who received PCCS '
                         f'after the cohort end date of {max_chemo_date}.\n')
        df = df[~mask1 & ~mask2]
        
    time = month_diff(df[cols[x]], df[cols[y]])
    if clip: time = time.clip(upper=time.quantile(q=0.999))
    time = time.sort_values()
    return time

def month_diff(d1, d2):
    """Get the months between datetimes"""
    avg_days_in_month = 30.437
    # return (d1.dt.year - d2.dt.year)*12 + d1.dt.month - d2.dt.month
    return (d1 - d2).dt.days / avg_days_in_month
            

###############################################################################
# Estimated Glomerular Filteration Rate
###############################################################################
def get_eGFR(df, col='value', prefix=''):
    """Estimate the glomerular filteration rate according to [1] for 
    determining chronic kidney disease

    Reference:
        [1] https://www.kidney.org/professionals/kdoqi/gfr_calculator/formula
    """
    for sex, group in df.groupby('sex'):
        params = eGFR_params[sex]
        
        # convert umol/L to mg/dL (1 umol/L = 0.0113 mg/dl)
        scr_count = group[col]*0.0113
        
        scr_term = scr_count/params['K']
        min_scr_term, max_scr_term = scr_term.copy(), scr_term.copy()
        min_scr_term[min_scr_term > 1] = 1
        max_scr_term[max_scr_term < 1] = 1
        eGFR = 142 * \
            (min_scr_term**params['a']) * \
            (max_scr_term**-1.2) * \
            (0.9938**group['age']) * \
            params['multiplier']
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
    
    # write the results
    filepath = f'{output_path}/tables/hyperparameters.csv'
    hyperparams.to_csv(filepath, index_label='index')
    
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
    
    # write the results
    filepath = f'{output_path}/tables/pearson_matrix.csv'
    pearson_matrix.to_csv(filepath, index_label='index')
    
    return pearson_matrix
