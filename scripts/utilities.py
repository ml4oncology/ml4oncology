import os
import pickle
import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from functools import partial
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from scripts.config import (root_path, regiments_folder, plus_minus, all_observations, cancer_code_mapping, 
                            nn_solvers, nn_activations, clean_variable_mapping)
from scripts.preprocess import (clean_string, split_and_parallelize)

twolevel = pd.MultiIndex.from_product([[], []])

# Calculate the Min/Max Expected (Reference Range) Observation Values
def get_observation_ranges():
    obs_ranges = {obs_code: [np.inf, 0] for obs_code in set(all_observation.values())}
    cols = ['ReferenceRange', 'ObservationCode']
    chunks = pd.read_csv(f"{root_path}/data/olis_complete.csv", chunksize=10**6)
    for i, chunk in tqdm.tqdm(enumerate(chunks)):
        # keep columns of interest
        chunk = chunk[cols].copy()
        # remove rows where no reference range is given
        chunk = chunk[~chunk['ReferenceRange'].isnull()]
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

# Retrieve selected regimens
def read_partially_reviewed_csv():
    df = open(f'{root_path}/{regiments_folder}/{regiments_folder}_regiments.csv')
    cols = next(df)
    cols = cols.strip().split(',')
    values = []
    for line in df:
        # make sure each line has correct number of entries
        line = line.strip().replace(',,', ',').split(',')
        if len(line) < len(cols): line.append('')
        if len(line) < len(cols): line.append('')
        if len(line) > len(cols): 
            new_note = ('').join(line[len(cols)-1:])
            line = line[:len(cols)-1]
            line.append(new_note)
        values.append(line)
    return pd.DataFrame(values, columns=cols)

def get_included_regimen(df):
    df = df[df['include (1) or exclude (0)']=='1'] 
    df = df.drop(columns=['include (1) or exclude (0)'])
    df = df.set_index('regiments')
    return df

# Load / Save
def load_ml_model(model_dir, algorithm):
    filename = f'{model_dir}/{algorithm}_classifier.pkl'
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

def load_ensemble_weights(save_dir, ml_models):
    filename = f'{save_dir}/ENS_classifier_best_param.pkl'
    with open(filename, 'rb') as file:
        ensemble_weights = pickle.load(file)
    ensemble_weights = [ensemble_weights[alg] for alg in ml_models] # ensure that the weight order matches ml_models
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
            print(f'No pos examples, skipping bootstrap #{i}')
            continue
        scores.append((roc_auc_score(y_true, y_pred), average_precision_score(y_true, y_pred)))
    return scores

def compute_bootstrap_scores(Y_true, Y_pred, n_bootstraps=10000, processes=32):
    worker = partial(bootstrap_worker, Y_true=Y_true, Y_pred=Y_pred)
    scores = split_and_parallelize(range(n_bootstraps), worker, processes=processes)
    return scores
        
# Post Training Analysis
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
    print(f'Desired {metric} {desired_target} could not be achieved. Closest {metric} achieved was {cur_target}')
    return -1

def most_common_by_category(data, category='regimen', top=5):
    # most common `category` (e.g. regimen, cancer location, cancer type) in terms of number of patients
    get_num_patients = lambda group: group['ikn'].nunique()
    num_ikn_by_category = data.groupby(category).apply(get_num_patients)
    most_common = num_ikn_by_category.sort_values(ascending=False)[0:top]
    return most_common

def get_clean_variable_names(cols):
    # swap cause and event
    for event in ['ED', 'H']:
        for cause in ['INFX', 'TR', 'GI']:
            cols = cols.str.replace(f'{cause}_{event}', f'{event}_{cause}')
    # mapping = list(clean_variable_mapping.items()) + list(cancer_code_mapping.items())
    mapping = list(clean_variable_mapping.items())
    for orig, new in mapping:
        cols = cols.str.replace(orig, new)
    cols = cols.str.replace('_', ' ')
    cols = cols.str.title()
    cols = cols.str.replace('Ed', 'ED')
    return cols

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
                num_nonimmigrants, num_immigrants = self.X.loc[self.patient_indices, 'is_immigrant'].value_counts()
            elif with_respect_to == 'sessions':
                num_nonimmigrants, num_immigrants = self.X['is_immigrant'].value_counts()
            summary_df.loc[('Immigrantion', 'Immigrant'), col] = f"{num_immigrants} ({np.round(num_immigrants/total*100, 1)}%)"
            summary_df.loc[('Immigrantion', 'Nonimmigrant'), col] = f"{num_nonimmigrants} ({np.round(num_nonimmigrants/total*100, 1)}%)"
            
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
    
    def get_summary(self, summary_df):
        self.num_sessions_per_patient_summary(summary_df)
        self.avg_age_summary(summary_df)
        self.immigration_summary(summary_df)
        self.sex_summary(summary_df)
        self.regimen_summary(summary_df, top=4)
        self.cancer_location_summary(summary_df, top=4)
        self.cancer_type_summary(summary_df, top=3)

def data_splits_summary(train, model_data, save_dir):
    # Train, Valid, Test split population summary
    summary_df = pd.DataFrame(index=twolevel, columns=twolevel)
    for split, (X, _) in train.data_splits.items():
        X = model_data.loc[X.index]
        dss = DataSplitSummary(X, split)
        dss.get_summary(summary_df)
        
    summary_df.to_csv(f'{save_dir}/data_splits_summary.csv')
    return summary_df

class SubgroupPerformanceSummary:
    def __init__(self, model_dir, algorithm, X, Y, data, target_types, preds, pred_thresh=0.2, display_ci=False, load_ci=False):
        self.model_dir = model_dir
        self.algorithm = algorithm
        self.model = None if algorithm == 'ENS' else load_ml_model(self.model_dir, self.algorithm)
        self.X = X
        self.Y = Y
        self.data = data
        self.target_types = target_types
        self.N = self.data['ikn'].nunique()
        self.preds = preds
        self.pred_thresh = pred_thresh
        self.display_ci = display_ci
        self.load_ci = load_ci
        self.ci_df = pd.DataFrame(index=twolevel)
        if self.load_ci:
            self.ci_df = pd.read_csv(f'{self.model_dir}/confidence_interval/bootstrapped_subgroup_scores_{algorithm}.csv', index_col=[0,1])
            self.ci_df.columns = self.ci_df.columns.astype(int)
    
    def predict(self, row_name, X):
        if row_name not in self.preds:
            pred = self.model.predict_proba(X) 
            pred =  pred.T if self.algorithm == 'NN' else np.array(pred)[:, :, 1]
            self.preds[row_name] = pred
        return self.preds[row_name]
    
    def get_bootstrapped_scores(self, Y_true, Y_pred, row_name, target_type, n_bootstraps=10000):
        group_name, subgroup_name = row_name
        group_name = group_name.lower().replace(' ', '_')
        subgroup_name = subgroup_name.split('(')[0].strip().lower().replace(' ', '_').replace('/', '_') # WARNING: SUPER HARD CODED
        ci_index = f'{group_name}_{subgroup_name}_{target_type}'
        if ci_index not in self.ci_df.index:
            auc_scores = compute_bootstrap_scores(Y_true.reset_index(drop=True), Y_pred, n_bootstraps=n_bootstraps)
            print(f'Completed bootstrap computations for {ci_index}')
            auroc_scores, auprc_scores = np.array(auc_scores).T
            self.ci_df.loc[(ci_index, 'AUROC'), range(n_bootstraps)] = auroc_scores
            self.ci_df.loc[(ci_index, 'AUPRC'), range(n_bootstraps)] = auprc_scores
        return self.ci_df.loc[ci_index].values
        
    def get_confidence_interval(self, Y_true, Y_pred, summary_df, row_name, target_type):
        auroc_scores, auprc_scores = self.get_bootstrapped_scores(Y_true, Y_pred, row_name, target_type)
        for name, scores in [('AUROC Score', auroc_scores), ('AUPRC Score', auprc_scores)]:
            confidence_interval = 1.96 * scores.std() / np.sqrt(len(scores))
            confidence_interval = np.round(confidence_interval * 100, 4) # convert it to percentage since number is so small
            col_name = (target_type, name)
            summary_df.loc[row_name, col_name] = f'{summary_df.loc[row_name, col_name]} {plus_minus} {confidence_interval}%'
        return summary_df

    def score_within_subgroups(self, row_name, X, Y, summary_df):
        pred_prob = self.predict(row_name, X)
        for idx, target_type in enumerate(self.target_types):
            Y_true = Y[target_type]
            Y_pred_prob = pred_prob[idx]
            if Y_true.nunique() < 2:
                print(f'No pos examples, skipping {target_type} - {row_name}')
                continue
            
            # AUROC/AUPRC Score
            summary_df.loc[row_name, (target_type, 'AUROC Score')] = np.round(roc_auc_score(Y_true, Y_pred_prob), 3)
            summary_df.loc[row_name, (target_type, 'AUPRC Score')] = np.round(average_precision_score(Y_true, Y_pred_prob), 3)
            if self.display_ci:
                summary_df = self.get_confidence_interval(Y_true, Y_pred_prob, summary_df, row_name, target_type)
                
            # PPV/Sensitivity @ pred_threshold=pred_threshold
            Y_pred_bool = Y_pred_prob > self.pred_thresh
            col_name =  (target_type, f'PPV @ Prediction Threshold={self.pred_thresh}')
            summary_df.loc[row_name, col_name] = np.round(precision_score(Y_true, Y_pred_bool, zero_division=1), 3)
            col_name =  (target_type, f'Sensitivity @ Prediction Threshold={self.pred_thresh}')
            summary_df.loc[row_name, col_name] = np.round(recall_score(Y_true, Y_pred_bool, zero_division=1), 3)
                
    def score_for_most_common_category_subgroups(self, title, category, summary_df, mapping=None):
        for cat_feature, num_patients in most_common_by_category(self.data, category=category, top=3).iteritems():
            X_subgroup = self.X[self.X[f'{category}_{cat_feature}'] == 1]
            Y_subgroup = self.Y.loc[X_subgroup.index] 
            if mapping: cat_feature = mapping[cat_feature]
            name = (title, f'{cat_feature} ({np.round(num_patients/self.N*100, 1)}%)')
            self.score_within_subgroups(name, X_subgroup, Y_subgroup, summary_df)

    def score_for_days_since_starting_subgroups(self, summary_df):
        col = 'days_since_starting_chemo'
        for (low, high) in [(0, 30), (31, 90), (91, np.inf)]:
            mask = (low <= self.data[col]) & (self.data[col] <= high)
            X_subgroup = self.X.loc[self.data[mask].index]
            Y_subgroup = self.Y.loc[X_subgroup.index] 
            interval = f'{low}+' if high == np.inf else f'{low}-{high}'
            name = ('Days Since Starting Regimen', interval)
            self.score_within_subgroups(name, X_subgroup, Y_subgroup, summary_df)

    def score_for_entire_test_set(self, summary_df):
        name = ('Entire Test Cohort', f'{self.N} patients (100%)')
        self.score_within_subgroups(name, self.X, self.Y, summary_df)
    
    def score_for_age_subgroups(self, summary_df):
        col = 'age'
        for (low, high) in [(18, 64), (65, np.inf)]:
            mask = (low <= self.data[col]) & (self.data[col] <= high)
            X_subgroup = self.X.loc[self.data[mask].index]
            Y_subgroup = self.Y.loc[X_subgroup.index] 
            num_patients = self.data.loc[mask, 'ikn'].nunique()
            interval = f'{low}+' if high == np.inf else f'{low}-{high}'
            name = ('Age',  f'{interval} ({np.round(num_patients/self.N*100, 1)}%)')
            self.score_within_subgroups(name, X_subgroup, Y_subgroup, summary_df)
            
    def score_for_sex_subgroups(self, summary_df):
        col = 'sex'
        for sex, name in [('F', 'Female'), ('M', 'Male')]:
            mask = self.data[col] == sex
            X_subgroup = self.X.loc[self.data[mask].index]
            Y_subgroup = self.Y.loc[X_subgroup.index] 
            num_patients = self.data.loc[mask, 'ikn'].nunique()
            name = ('Sex',  f'{name} ({np.round(num_patients/self.N*100, 1)}%)')
            self.score_within_subgroups(name, X_subgroup, Y_subgroup, summary_df)
            
    def score_for_immigrant_subgroups(self, summary_df):
        for mask_bool, col_name, col in [(True, 'Immigrant', 'is_immigrant'), 
                                         (False, 'Non-immigrant', 'is_immigrant'),
                                         (True, 'English Speaker', 'speaks_english'),
                                         (False, 'Non-English Speaker', 'speaks_english')]:
            mask = self.data[col] if mask_bool else ~self.data[col]
            X_subgroup = self.X.loc[self.data[mask].index]
            Y_subgroup = self.Y.loc[X_subgroup.index] 
            num_patients = self.data.loc[mask, 'ikn'].nunique()
            col = col.replace('_', ' ').title()
            name = ('Immigration',  f'{col_name} ({np.round(num_patients/self.N*100, 1)}%)')
            self.score_within_subgroups(name, X_subgroup, Y_subgroup, summary_df)
            
    def get_summary(self, summary_df):
        self.score_for_entire_test_set(summary_df)
        self.score_for_age_subgroups(summary_df)
        self.score_for_sex_subgroups(summary_df)
        self.score_for_immigrant_subgroups(summary_df)
        self.score_for_most_common_category_subgroups('Regimen', 'regimen', summary_df)
        self.score_for_most_common_category_subgroups('Cancer Location', 'curr_topog_cd', summary_df, mapping=cancer_code_mapping)
        self.score_for_days_since_starting_subgroups(summary_df)
            
def subgroup_performance_summary(save_dir, algorithm, model_data, train, target_types=None, save_preds=True, load_preds=False, 
                                 display_ci=False, load_ci=False, save_ci=False):
    """
    Args:
        display_ci (bool): display confidence interval
        load_ci (bool): load saved scores for computing confidence interval or recomuputing the bootstrapped scores
    """
    if target_types is None:
        target_types = train.target_types
        
    preds = {}
    if load_preds:
        preds = load_predictions(f'{save_dir}/predictions', filename=f'subgroup_predictions_{algorithm}')
        
    summary_df = pd.DataFrame(index=twolevel, columns=twolevel)
    data = model_data.loc[train.X_test.index]
    sps = SubgroupPerformanceSummary(save_dir, algorithm, train.X_test, train.Y_test, data, target_types, preds, 
                                     display_ci=display_ci, load_ci=load_ci)
    sps.get_summary(summary_df)
    
    if save_preds: save_predictions(sps.preds, f'{save_dir}/predictions', filename=f'subgroup_predictions_{algorithm}')
    if save_ci: sps.ci_df.to_csv(f'{save_dir}/confidence_interval/bootstrapped_subgroup_scores_{algorithm}.csv')
        
    summary_df.to_csv(f'{save_dir}/tables/subgroup_performance_summary_{algorithm}.csv')
    return summary_df

# Get hyperparams
def get_hyperparameters(main_dir, days=30):
    hyperparams = pd.DataFrame(index=twolevel, columns=['Hyperparameter Value'])
    for algorithm in ['LR', 'RF', 'XGB', 'NN', 'ENS', 'GRU']:
        extra_path = f'GRU/within_{days}_days/hyperparam_tuning/' if algorithm == 'GRU' else f'ML/within_{days}_days/best_params/'
        filename = f'{main_dir}/models/{extra_path}{algorithm}_classifier_best_param.pkl'
        with open(filename, 'rb') as file:
            best_param = pickle.load(file)
        for param, value in best_param.items():
            if param == 'solver': value = nn_solvers[int(value)].upper()
            if param == 'activation': value = nn_activations[int(value)].upper()
            param = param.replace('_', ' ').title()
            if algorithm == 'ENS': param = f'{param.upper()} Weight'
            hyperparams.loc[(algorithm, param), 'Hyperparameter Value'] = value
    hyperparams.to_csv(f'{main_dir}/models/hyperparameters.csv', index_label='index')
    return hyperparams

# Plot Functions
def remove_top_right_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
def get_bbox(ax, fig, pad_x0=0.75, pad_y0=0.5, pad_x1=0.1, pad_y1=0.1):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    bbox.x0 -= pad_x0
    bbox.y0 -= pad_y0
    bbox.x1 += pad_x1
    bbox.y1 += pad_y1
    return bbox
    
def num_patients_per_regimen(df):
    # number of patients per cancer regiment
    regiments = []
    patients = []
    for regimen, group in df.groupby('regimen'):
        regiments.append(regimen)
        patients.append(len(group['ikn'].unique()))
    plt.bar(regiments, patients) 
    plt.xlabel('Chemotherapy Regiments')
    plt.ylabel('Number of Patients')
    plt.xticks(rotation=90)

def num_blood_counts_per_regimen(df):
    # number of blood counts per regimen
    regiments = []
    blood_counts = []
    for regimen, group in df.groupby('regimen'):
        regiments.append(regimen)
        blood_counts.append(sum((~group[range(-5,29)].isnull()).sum(axis=1)))
    plt.bar(regiments, blood_counts) 
    plt.xlabel('Chemotherapy Regiments')
    plt.ylabel('Number of Blood Measurements')
    plt.xticks(rotation=90)

def hist_blood_counts(df):
    # histogram of numbers of blood counts measured (for a single sample)
    df = df[range(-5,29)]
    blood_counts = (~df[range(-5,29)].isnull()).sum(axis=1).values
    n, bins, patches = plt.hist(blood_counts, bins=33)
    plt.title('Histogram of Number of Blood Measurements for a Single Sample')

def hist_days_btwn_prev_and_last_count(df):
    # histogram of number of days between the last and the previous blood count measurement (for a single row)
    get_num_days_between_mes = lambda row: np.diff(np.where(~np.isnan(row))[0][-2:])
    values = df[range(-5,29)].values
    days_in_between = np.array([get_num_days_between_mes(row) for row in values])                   
    rows_per_days_in_between = [sum(days_in_between < day)[0] for day in range(2, 34)]
    plt.bar(range(1,33), rows_per_days_in_between)
    plt.title('Histogram of number of days between last and prev blood measurement')
    plt.show()

def scatter_plot(df, cycle_lengths, unit='10^9/L', save=False, filename="NEUTROPHIL_PLOT1"):
    num_regimen = len(df['regimen'].unique())
    fig = plt.figure(figsize=(15,150))
    for idx, (regimen, group) in tqdm.tqdm(enumerate(df.groupby('regimen'))):
        cycle_length = int(cycle_lengths[regimen])
        y = group[range(0,cycle_length+1)].values.flatten()
        x = np.array(list(range(0,cycle_length+1))*len(group))

        ax = fig.add_subplot(num_regimen,2,idx+1)
        plt.subplots_adjust(hspace=0.3)
        plt.scatter(x, y, alpha=0.03)
        plt.title(regimen)
        plt.ylabel(f'Blood Count ({unit})')
        plt.xlabel('Day')
        if cycle_length < 15: plt.xticks(range(0, cycle_length+1))
    if save: 
        plt.savefig(f'{root_path}/cytopenia/plots/{filename}.jpg', bbox_inches='tight', dpi=300)
    plt.show()

def below_threshold_bar_plot(df, cycle_lengths, threshold, save=False, filename='NEUTROPHIL_PLOT2', color=None):
    num_regimen = len(set(df['regimen']))
    fig = plt.figure(figsize=(15, 150))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    idx = 1
    for regimen, group in tqdm.tqdm(df.groupby('regimen')):
        # print(regimen)
        # print("Ratio of blood count versus nans (how sparse the dataframe is) =", (~group[range(-5,28)].isnull()).values.sum() / (len(group)*33))
        cycle_length = int(cycle_lengths[regimen])
        num_patients = (~group[range(0,cycle_length+1)].isnull()).sum(axis=0).values
        # c annot display data summary with observations less than 6, replace them with 6
        num_patients[(num_patients < 6) & (num_patients > 0)] = 6
        ax = fig.add_subplot(num_regimen,3,idx)
        plt.bar(range(0,cycle_length+1), num_patients, color=color)
        plt.title(regimen)
        plt.ylabel('Number of patients')
        plt.xlabel('Day')
        if cycle_length < 15: plt.xticks(range(0, cycle_length+1))
  
        num_patient_below_threshold = np.array([len(group.loc[group[day] < threshold, 'ikn'].unique()) for day in range(0,cycle_length+1)])
        # cannot display data summary with observations less than 6, replace them with 6
        num_patient_below_threshold[(num_patient_below_threshold < 6) & (num_patient_below_threshold > 0)] = 6
        ax = fig.add_subplot(num_regimen,3,idx+1)
        plt.bar(range(0,cycle_length+1), num_patient_below_threshold, color=color)
        plt.title(regimen)
        plt.ylabel(f'Number of patients\nwith blood count < {threshold}')
        plt.xlabel('Day')
        if cycle_length < 15: plt.xticks(range(0, cycle_length+1))
  
        num_patient_below_threshold = np.array([len(group.loc[group[day] < threshold, 'ikn'].unique()) for day in range(0,cycle_length+1)])
        ax = fig.add_subplot(num_regimen,3,idx+2)
        plt.bar(range(0,cycle_length+1), num_patient_below_threshold/num_patients, color=color)
        plt.title(regimen)
        plt.ylabel(f'Percentage of patients\nwith blood count < {threshold}')
        plt.xlabel('Day')
        if cycle_length < 15: plt.xticks(range(0, cycle_length+1))
  
        idx += 3
    plt.text(0, -0.3, '*Observations < 6 are displayed as 6', transform=ax.transAxes, fontsize=12)
    if save:
        plt.savefig(f'{root_path}/cytopenia/plots/{filename}.jpg', bbox_inches='tight', dpi=300)
    plt.show()

def iqr_plot(df, cycle_lengths, unit='10^9/L', show_outliers=True, save=False, filename='NEUTROPHIL_PLOT3',
             figsize=(15,150)):
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=0.3)
    nadir_dict = {}
    num_regimen = len(set(df['regimen']))
    for idx, (regimen, group) in tqdm.tqdm(enumerate(df.groupby('regimen'))):
        cycle_length = int(cycle_lengths[regimen])
        data = np.array([group[day].dropna().values for day in range(0,cycle_length+1)], dtype=object)
        # or group[range(-5,29)].boxplot()
        ax = fig.add_subplot(num_regimen,2,idx+1)
        bp = plt.boxplot(data, labels=range(0,cycle_length+1), showfliers=show_outliers)
        plt.title(regimen)
        plt.ylabel(f'Blood Count ({unit})')
        plt.xlabel('Day')
    
        medians = [median.get_ydata()[0] for median in bp['medians']]
        min_idx = np.nanargmin(medians)
        nadir_dict[regimen] = {'Day of Nadir': min_idx-5, 'Depth of Nadir (Min Blood Count)': medians[min_idx]}
    
        plt.plot(range(1,cycle_length+2), medians, color='red')

    if save:
        plt.savefig(f'{root_path}/cytopenia/plots/{filename}.jpg', bbox_inches='tight', dpi=300)    
    plt.show()
    nadir_df = pd.DataFrame(nadir_dict)
    return nadir_df.T

def mean_cycle_plot(df, cycle_lengths, unit='10^9/L', save=False, filename='NEUTROPHIL_PLOT4'):
    fig = plt.figure(figsize=(15, 150))
    plt.subplots_adjust(hspace=0.3)
    cycles = [1,2,3,4,5]
    num_regimen = len(set(df['regimen']))
    for idx, (regimen, group) in tqdm.tqdm(enumerate(df.groupby('regimen'))):
        cycle_length = int(cycle_lengths[regimen])
        ax = fig.add_subplot(num_regimen,2,idx+1)
    
        for cycle in cycles:
            tmp_df = group[group['chemo_cycle'] == cycle]
            medians = tmp_df[range(0,cycle_length+1)].median().values
            plt.plot(range(0, cycle_length+1), medians)
    
        plt.title(regimen)
        plt.ylabel(f'Median Blood Count ({unit})')
        plt.xlabel('Day')
        plt.legend([f'cycle{c}' for c in cycles])
        if cycle_length < 15: plt.xticks(range(0, cycle_length+1))
    if save:
        plt.savefig(f'{root_path}/cytopenia/plots/{filename}.jpg', bbox_inches='tight', dpi=300)
    plt.show()
    
def feat_importance_plot(algorithm, target_types, save_dir, figsize, top=20):
    # NOTE: run `python scripts/perm_importance.py` in the command line before running this function
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=0.3)
    N = len(target_types)
    
    df = pd.read_csv(f'{save_dir}/perm_importance/{algorithm}.csv')
    df = df.set_index('index')
    df.index = get_clean_variable_names(df.index)
    
    for idx, target_type in tqdm.tqdm(enumerate(target_types)):
        feature_importances = df[target_type]
        feature_importances = feature_importances.sort_values(ascending=False)
        feature_importances = feature_importances[0:top] # get the top important features
        ax = fig.add_subplot(N,1,idx+1)
        ax.barh(feature_importances.index, feature_importances.values)
        ax.invert_yaxis()
        ax.set_xlabel('Permutation Importance Score')
        remove_top_right_axis(ax)
        fig.savefig(f'{save_dir}/figures/important_features_{algorithm}_{target_type}.jpg', 
                    bbox_inches=get_bbox(ax, fig, pad_x0=3.7), dpi=300) 
        ax.set_title(target_type) # set title AFTER saving individual figures
    plt.savefig(f'{save_dir}/figures/important_features.jpg', bbox_inches='tight', dpi=300)
    
def subgroup_performance_plot(df, save_dir, target_type='ACU', name='Acute Care', figsize=(16,18)):
    
    def get_score(subgroup_name, metric):
        tmp = df.loc[subgroup_name, (target_type, metric)]
        tmp = tmp.str.split(plus_minus)
        return tmp.str[0].astype(float)
    
    # Get the bar names
    bar_names = df.index.levels[1]
    bar_names = bar_names.str.split(pat='(').str[0]
    bar_names = bar_names.str.strip().str.title()
    bar_names = bar_names.tolist()
    bar_names[0] = df.index.levels[0][0]

    # Bar plot
    start_pos = 0
    x_bar_pos = []
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=figsize)
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.6)
    metrics = df.columns.levels[1]
    for idx, subgroup_name in enumerate(df.index.levels[0]):
        auroc_scores = get_score(subgroup_name, metrics[0]) # AUROC Score
        bar_pos = np.arange(start_pos, start_pos+len(auroc_scores)).tolist()
        x_bar_pos += bar_pos
        start_pos += len(auroc_scores) + 1
        axes[0].bar(bar_pos, auroc_scores, label=subgroup_name, width=0.8)
   
        auprc_scores = get_score(subgroup_name, metrics[1]) # AUPRC Score
        axes[1].bar(bar_pos, auprc_scores, label=subgroup_name, width=0.8)
        
        ppv_scores = df.loc[subgroup_name, (target_type, metrics[2])] # PPV
        axes[2].bar(bar_pos, ppv_scores, label=subgroup_name, width=0.8)
        
        sensitivity_scores = df.loc[subgroup_name, (target_type, metrics[3])] # Sensitivity
        axes[3].bar(bar_pos, sensitivity_scores, label=subgroup_name, width=0.8)
        
        if idx == 0:
            param = {'color': 'black', 'linestyle': '--'}
            axes[0].axhline(y=auroc_scores[0], **param)
            axes[1].axhline(y=auprc_scores[0], **param)
            axes[2].axhline(y=ppv_scores[0], **param)
            axes[3].axhline(y=sensitivity_scores[0], **param)
        
    for i, metric in enumerate(metrics):
        remove_top_right_axis(axes[i])
        axes[i].set_xticks(x_bar_pos) # adjust x-axis position of bars # Alternative way: plt.xticks(x_bar_pos, bar_names, rotation=45)
        axes[i].set_xticklabels(bar_names, rotation=45)
        axes[i].set_ylabel(metric)
        axes[i].legend(bbox_to_anchor=(1,0), loc='lower left', frameon=False)
        metric_name = metric.split('@')[0].strip()
        fig.savefig(f'{save_dir}/subgroup_performance_{target_type}_{metric_name}.jpg', 
                    bbox_inches=get_bbox(axes[i], fig, pad_y0=1.2, pad_x1=2.6, pad_y1=0.2), dpi=300) 
    plt.savefig(f'{save_dir}/subgroup_performance.jpg', bbox_inches='tight', dpi=300)