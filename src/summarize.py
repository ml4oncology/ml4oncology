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
"""
Module to create various summary tables
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import (average_precision_score, precision_score, recall_score, roc_auc_score)
from src.config import (cytopenia_gradings, symptom_cols, cancer_code_mapping, 
                        variable_groupings_by_keyword)
from src.utility import (twolevel, split_and_parallelize, 
                         compute_bootstrap_scores, nadir_bootstrap_worker,
                         get_clean_variable_names, get_cyto_rates, most_common_by_category)

###############################################################################
# Data Characteristics
###############################################################################
class DataPartitionSummary:
    """
    Computes total number of patients and treatment sessions in different population subgroups of a data partition
    """
    def __init__(self, X, Y, partition, top_category_items=None):
        """
        Args:
            X (pd.DataFrame): data partition in its original form (not one-hot encoded, normalized, clipped)
            Y (pd.DataFrame): target labels
            partition (string): name of the data partition e.g. All, Testing
            top_category_items (dict): mapping of categories and their top entries
        """
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
        self.display = lambda x, total: f"{x} ({x/total*100:.1f}%)"
        
    def num_sessions_per_patient_summary(self, summary_df):
        num_sessions = self.X.groupby('ikn').apply(len)
        mean = num_sessions.mean()
        std = num_sessions.std()
        summary_df.loc[('Number of Sessions', ''), self.col['patients']] = f'{mean:.1f}, SD ({std:.1f})'

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
                counts = self.X.loc[self.patient_indices, 'sex'].value_counts()
            elif with_respect_to == 'sessions':
                counts = self.X['sex'].value_counts()
            summary_df.loc[('Sex', 'Female'), col] = self.display(counts['F'], total)
            summary_df.loc[('Sex', 'Male'), col] = self.display(counts['M'], total)
            
    def income_summary(self, summary_df):
        name = 'neighborhood_income_quintile' # variable name
        title = name.replace('_', ' ').title()
        
        # do not include income quintile data that were imputed
        X = self.X[~self.X[f'{name}_is_missing']]
        
        for with_respect_to, col in self.col.items():
            total = self.total[with_respect_to]
            if with_respect_to == 'patients':
                mask = X.index.isin(self.patient_indices)
                counts = X.loc[mask, name].value_counts()
            elif with_respect_to == 'sessions':
                counts = X[name].value_counts()
                
            for income_quintile, count in counts.sort_index().iteritems():
                summary_df.loc[(title, f'Q{int(income_quintile)}'), col] = self.display(count, total)
            
    def immigration_summary(self, summary_df):
        for with_respect_to, col in self.col.items():
            total = self.total[with_respect_to]
            if with_respect_to == 'patients':
                immigrant_count = self.X.loc[self.patient_indices, 'is_immigrant'].value_counts()
                eng_speaker_count = self.X.loc[self.patient_indices, 'speaks_english'].value_counts()
            elif with_respect_to == 'sessions':
                immigrant_count = self.X['is_immigrant'].value_counts()
                eng_speaker_count = self.X['speaks_english'].value_counts()
            summary_df.loc[('Immigration', 'Immigrant'), col] = self.display(immigrant_count[True], total)
            summary_df.loc[('Immigration', 'Non-immigrant'), col] = self.display(immigrant_count[False], total)
            summary_df.loc[('Immigration', 'English Speaker'), col] = self.display(eng_speaker_count[True], total)
            summary_df.loc[('Immigration', 'Non-English Speaker'), col] = self.display(eng_speaker_count[False], total)
            
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
        # summarize the items not in top_items
        self.other_summary(summary_df, category, display_name, top_items)
        
    def combordity_summary(self, summary_df):
        for with_respect_to, col in self.col.items():
            total = self.total[with_respect_to]
            if with_respect_to == 'patients':
                diabetes_count = self.X.loc[self.patient_indices, 'diabetes'].value_counts()
                ht_count = self.X.loc[self.patient_indices, 'hypertension'].value_counts()
            elif with_respect_to == 'sessions':
                diabetes_count = self.X['diabetes'].value_counts()
                ht_count = self.X['hypertension'].value_counts()
            summary_df.loc[('Combordity', 'Diabetes'), col] = self.display(diabetes_count[True], total)
            summary_df.loc[('Combordity', 'Non-diabetes'), col] = self.display(diabetes_count[False], total)
            summary_df.loc[('Combordity', 'Hypertension'), col] = self.display(ht_count[True], total)
            summary_df.loc[('Combordity', 'Non-hypertension'), col] = self.display(ht_count[False], total)
    
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
        self.income_summary(summary_df)
        self.sex_summary(summary_df)
        self.category_subgroup_summary(summary_df, category='regimen', top=top)
        self.category_subgroup_summary(summary_df, category='curr_topog_cd', top=top)
        self.category_subgroup_summary(summary_df, category='curr_morph_cd', top=top)
        if include_target: self.target_summary(summary_df)
        if include_combordity: self.combordity_summary(summary_df)
        if include_gcsf: self.gcsf_summary(summary_df)
        if include_ckd: self.ckd_summary(summary_df)
        if include_dialysis: self.dialysis_summary(summary_df)

def data_characteristic_summary(eval_models, save_dir, partition='split', target_event=None, **kwargs):
    """
    Get characteristics summary of patients and treatments for each 
    data split (Train-Valid-Test) or cohort (Development-Testing)
    
    Development cohort refers to Training and Validation split
    
    Args:
        partition (str): how to partition the data for summarization, either by split or cohort
        target_event (None or str): get summarization for the portion of data in which target event has occured
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
    
    # Data target event occured summary
    if target_event is not None:
        X = X[Y[target_event]]
        Y = Y.loc[X.index]
        dps = DataPartitionSummary(X, Y, target_event, top_category_items=top_category_items)
        dps.get_summary(summary_df, **kwargs)
    
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

def top_cancer_regimen_summary(data, top=10):
    result = pd.DataFrame()
    top_cancers = most_common_by_category(data, category='curr_topog_cd', top=top) # cancer locations
    for cancer_code, nsessions in top_cancers.items():
        top_regimens = most_common_by_category(data[data['curr_topog_cd'] == cancer_code], category='regimen', top=top)
        top_regimens = [f'{regimen} ({num_sessions})' for regimen, num_sessions in top_regimens.items()]
        result[f'{cancer_code_mapping[cancer_code]} (N={nsessions})'] = top_regimens
    result.index += 1
    return result

###############################################################################
# Input Features
###############################################################################
def feature_summary(eval_models, prep, target_keyword, save_dir, deny_old_survey=True):
    """
    Args:
        deny_old_survey (bool): considers survey (symptom questionnaire) response more than 
                                5 days prior to chemo session as "missing" data
    """
    df = prep.dummify_data(eval_models.orig_data.copy())
    train_idxs = eval_models.labels['Train'].index
    df = df.loc[train_idxs]
    N = len(df)

    # remove missingness features, targets, first visit date, and ikn
    cols = df.columns
    df = df[cols[~(cols.str.contains('is_missing') | cols.str.contains(target_keyword))].drop('ikn')]

    # get mean, SD, and number of missing values for each feature for the training set
    summary = df.astype(float).describe()
    summary = summary.loc[['count', 'mean', 'std']].T
    summary = summary.round(6)
    summary['count'] = len(train_idxs) - summary['count']
    
    if deny_old_survey:
        event_dates = prep.event_dates.loc[train_idxs]
        for col in symptom_cols:
            if col in summary.index:
                days_before_chemo = event_dates['visit_date'] - event_dates[f'{col}_survey_date']
                summary.loc[col, 'count'] = sum(days_before_chemo > pd.Timedelta('5 days'))
    
    format_arr = lambda arr: arr.round(2).astype(str)
    summary['Missingness (%)'] = (summary['count'] / N * 100).round(1)
    summary['Mean (SD)'] = format_arr(summary['mean']) + ' (' + format_arr(summary['std']) + ')'
    summary = summary.rename(columns={'count': f'Train (N={N}) - Missingness Count', 
                                      'mean': 'Train - Mean', 
                                      'std': 'Train - SD'})

    # assign the groupings for each feature
    features = summary.index
    for group, keyword in variable_groupings_by_keyword.items():
        summary.loc[features.str.contains(keyword), 'Group'] = group
    
    # insert units
    rename_map = {name: f'{name} ({unit})' for name, unit in get_units().items()}
    summary = summary.rename(index=rename_map)
    
    summary.index = get_clean_variable_names(summary.index)
    summary.to_csv(f'{save_dir}/feature_summary.csv')
    return summary

###############################################################################
# Nadir
###############################################################################
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
    for regimen, group in tqdm(df.groupby('regimen')):
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

###############################################################################
# Subgroup Performance
###############################################################################
class SubgroupPerformanceSummary:
    def __init__(self, algorithm, eval_models, target_events,
                 pred_thresh=0.2, split='Test', display_ci=False, load_ci=False):
        self.model_dir = eval_models.output_path
        self.algorithm = algorithm
        self.split = split
        self.Y = eval_models.labels[split]
        self.entire_data = eval_models.orig_data # used for getting most common category subgroups
        self.data = self.entire_data.loc[self.Y.index]
        self.N = self.data['ikn'].nunique()
        self.target_events = target_events
        self.preds = eval_models.preds[split][algorithm]
        self.pred_thresh = pred_thresh
        self.display_ci = display_ci
        self.load_ci = load_ci
        self.ci_df = pd.DataFrame(index=twolevel)
        if self.load_ci:
            self.ci_df = pd.read_csv(f'{self.model_dir}/confidence_interval/bootstrapped_subgroup_scores_{algorithm}.csv', index_col=[0,1])
            self.ci_df.columns = self.ci_df.columns.astype(int)
    
    def get_bootstrapped_scores(self, Y_true, Y_pred, row_name, target_event, n_bootstraps=10000):
        group_name, subgroup_name = row_name
        group_name = group_name.lower().replace(' ', '_')
        subgroup_name = subgroup_name.split('(')[0].strip().lower().replace(' ', '_').replace('/', '_') # WARNING: SUPER HARD CODED
        ci_index = f'{group_name}_{subgroup_name}_{target_event}'
        if ci_index not in self.ci_df.index:
            auc_scores = compute_bootstrap_scores(Y_true.reset_index(drop=True), Y_pred, n_bootstraps=n_bootstraps)
            logging.info(f'Completed bootstrap computations for {ci_index}')
            auroc_scores, auprc_scores = np.array(auc_scores).T
            self.ci_df.loc[(ci_index, 'AUROC'), range(n_bootstraps)] = auroc_scores
            self.ci_df.loc[(ci_index, 'AUPRC'), range(n_bootstraps)] = auprc_scores
        return self.ci_df.loc[ci_index].values
        
    def get_confidence_interval(self, Y_true, Y_pred, summary_df, row_name, target_event):
        auroc_scores, auprc_scores = self.get_bootstrapped_scores(Y_true, Y_pred, row_name, target_event)
        for name, scores in [('AUROC', auroc_scores), ('AUPRC', auprc_scores)]:
            lower, upper = np.percentile(scores, [2.5, 97.5]).round(3)
            col_name = (target_event, name)
            summary_df.loc[row_name, col_name] = f'{summary_df.loc[row_name, col_name]} ({lower}-{upper})'
        return summary_df

    def score_within_subgroups(self, Y, row_name, summary_df):
        pred_prob = self.preds.loc[Y.index]
        for idx, target_event in enumerate(self.target_events):
            Y_true = Y[target_event]
            Y_pred_prob = pred_prob[target_event]
            pred_thresh = self.pred_thresh if isinstance(self.pred_thresh, float) else self.pred_thresh[idx]
            Y_pred_bool = Y_pred_prob > pred_thresh
            if Y_true.nunique() < 2:
                logging.warning(f'No pos examples, skipping {target_event} - {row_name}')
                continue
            
            # AUROC/AUPRC
            summary_df.loc[row_name, (target_event, 'AUROC')] = np.round(roc_auc_score(Y_true, Y_pred_prob), 3)
            summary_df.loc[row_name, (target_event, 'AUPRC')] = np.round(average_precision_score(Y_true, Y_pred_prob), 3)
            if self.display_ci:
                summary_df = self.get_confidence_interval(Y_true, Y_pred_prob, summary_df, row_name, target_event)
                
            # PPV/Sensitivity
            summary_df.loc[row_name, (target_event, 'PPV')] = np.round(precision_score(Y_true, Y_pred_bool, zero_division=1), 3)
            summary_df.loc[row_name, (target_event, 'Sensitivity')] = np.round(recall_score(Y_true, Y_pred_bool, zero_division=1), 3)
            
            # Warning Rate/Event Rate
            # summary_df.loc[row_name, (target_event, 'Warning Rate')] = np.round(Y_pred_bool.mean(), 3)
            summary_df.loc[row_name, (target_event, 'Event Rate')] = np.round(Y_true.mean(), 3)
                
    def score_for_most_common_category_subgroups(self, title, category, summary_df, mapping=None):
        for cat_feature, num_sessions in most_common_by_category(self.entire_data, category=category, top=3).items():
            mask = self.data[category] == cat_feature
            Y_subgroup = self.Y[mask] 
            num_patients = self.data.loc[mask, 'ikn'].nunique()
            if mapping: cat_feature = mapping[cat_feature]
            name = (title, f'{cat_feature} ({num_patients/self.N*100:.1f}%)')
            self.score_within_subgroups(Y_subgroup, name, summary_df)

    def score_for_days_since_starting_subgroups(self, summary_df):
        for (low, high) in [(0, 30), (31, 90), (91, np.inf)]:
            mask = self.data['days_since_starting_chemo'].between(low, high)
            Y_subgroup = self.Y[mask]
            interval = f'{low}+' if high == np.inf else f'{low}-{high}'
            name = ('Days Since Starting Regimen', interval)
            self.score_within_subgroups(Y_subgroup, name, summary_df)

    def score_for_entire_test_set(self, summary_df):
        name = (f'Entire {self.split} Cohort', f'{self.N} patients (100%)')
        self.score_within_subgroups(self.Y, name, summary_df)
    
    def score_for_age_subgroups(self, summary_df):
        for (low, high) in [(18, 64), (65, np.inf)]:
            mask = self.data['age'].between(low, high)
            Y_subgroup = self.Y[mask]
            num_patients = self.data.loc[mask, 'ikn'].nunique()
            interval = f'{low}+' if high == np.inf else f'{low}-{high}'
            name = ('Age',  f'{interval} ({num_patients/self.N*100:.1f}%)')
            self.score_within_subgroups(Y_subgroup, name, summary_df)
            
    def score_for_sex_subgroups(self, summary_df):
        mapping = {'F': 'Female', 'M': 'Male'}
        for sex, group in self.data.groupby('sex'):
            Y_subgroup = self.Y.loc[group.index]
            num_patients = group['ikn'].nunique()
            name = ('Sex',  f'{mapping[sex]} ({num_patients/self.N*100:.1f}%)')
            self.score_within_subgroups(Y_subgroup, name, summary_df)
            
    def score_for_immigrant_subgroups(self, summary_df):
        for mask_bool, col_name, col in [(True, 'Immigrant', 'is_immigrant'), 
                                         (False, 'Non-immigrant', 'is_immigrant'),
                                         (True, 'English Speaker', 'speaks_english'),
                                         (False, 'Non-English Speaker', 'speaks_english')]:
            mask = self.data[col] if mask_bool else ~self.data[col]
            Y_subgroup = self.Y[mask]
            num_patients = self.data.loc[mask, 'ikn'].nunique()
            name = ('Immigration',  f'{col_name} ({num_patients/self.N*100:.1f}%)')
            self.score_within_subgroups(Y_subgroup, name, summary_df)
            
    def score_for_income_subgroups(self, summary_df):
        for income_quintile, group in self.data.groupby('neighborhood_income_quintile'):
            Y_subgroup = self.Y.loc[group.index]
            num_patients = group['ikn'].nunique()
            name = ('Neighborhood Income Quintile',  f'Q{int(income_quintile)} ({num_patients/self.N*100:.1f}%)')
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
        if 'income' in subgroups: self.score_for_income_subgroups(summary_df)
        if 'regimen' in subgroups: self.score_for_most_common_category_subgroups('Regimen', 'regimen', summary_df)
        if 'cancer_location' in subgroups: 
            self.score_for_most_common_category_subgroups('Cancer Location', 'curr_topog_cd', summary_df, mapping=cancer_code_mapping)
        if 'days_since_starting' in subgroups: self.score_for_days_since_starting_subgroups(summary_df)
        if 'cycle_length' in subgroups: self.score_for_cycle_length_subgroups(summary_df)
        if 'ckd' in subgroups: self.score_for_ckd_subgroups(summary_df)
            
def subgroup_performance_summary(algorithm, eval_models, pred_thresh=0.2, subgroups=None, target_events=None, 
                                 display_ci=False, load_ci=False, save_ci=False):
    """
    Args:
        display_ci (bool): display confidence interval
        load_ci (bool): load saved scores for computing confidence interval or recomuputing the bootstrapped scores
        pred_thresh (float or array-like): prediction threshold for alarm trigger, can be provided as a single float value
                                           used for all models, or array of float values used for each model
    """
    if target_events is None:
        target_events = eval_models.target_events
        
    if subgroups is None:
        subgroups = {'all', 'age', 'sex', 'immigrant', 'income', 'regimen', 'cancer_location', 'days_since_starting'}
        
    summary_df = pd.DataFrame(index=twolevel, columns=twolevel)
    sps = SubgroupPerformanceSummary(algorithm, eval_models, target_events, 
                                     pred_thresh=pred_thresh, display_ci=display_ci, load_ci=load_ci)
    sps.get_summary(subgroups, summary_df)
    
    save_dir = eval_models.output_path
    if save_ci: 
        sps.ci_df.to_csv(f'{save_dir}/confidence_interval/bootstrapped_subgroup_scores_{algorithm}.csv')
    summary_df.to_csv(f'{save_dir}/tables/subgroup_performance_summary_{algorithm}.csv')
    return summary_df

def get_worst_performing_subgroup(eval_models, category='regimen', algorithm='XGB', target_event='Neutropenia', split='Valid'):
    """analyze subgroups with the worst performance (usually in the validation set)"""
    summary_df = pd.DataFrame(columns=twolevel)
    sps = SubgroupPerformanceSummary(algorithm, eval_models, target_events=[target_event], split=split)
    for regimen, group in sps.data.groupby(category):
        Y_subgroup = sps.Y.loc[group.index]
        sps.score_within_subgroups(Y_subgroup, regimen, summary_df)
    summary_df[(target_event, 'NSessions')] = sps.data[category].value_counts()
    summary_df = summary_df.sort_values(by=(target_event, 'AUROC'))
    summary_df.index = [cancer_code_mapping.get(index, index) for index in summary_df.index]
    return summary_df

###############################################################################
# Palliative Consultation Service
###############################################################################
def serivce_request_summary(eval_models, event_dates, thresholds, target_event,
                            days_ago=730, by_patients=True, algorithm='ENS', split='Test'):
    """
    Get number of sessions or patients in which early palliative consultation service was requested,
    for both predicted outcomes (e.g. Will experience target event vs Won't experience target event)
    
    For patients, we take the first session in which model triggered an alarm (risk prediction > threshold), 
    and determine if palliative consultation service was requested prior to that sesssion.
    If model never triggered an alarm, we use the last session of that patient.
    
    Args:
        thresholds (array-like): prediction thresholds to evaluate on
        days_ago (int): number of days prior to treatment a patient requests 
                        palliative consultation service to be considered timely
        by_patients (bool): summary of service requests in terms of number of patients
    """
    df = pd.DataFrame()
    # Get prediction (risk) of target event
    df['pred_prob'] = eval_models.preds[split][algorithm][target_event]
    
    # Determine if patient requested palliative consultation service within X days prior to treatment
    service_date = event_dates['palliative_consultation_service_date']
    visit_date = event_dates['visit_date']
    df['requested_service'] = (visit_date - service_date).dt.days <= days_ago
    
    if by_patients:
        # Get patient id 
        df['ikn'] = eval_models.orig_data['ikn']
    
    matrices = []
    for pred_thresh in thresholds:
        df['pred_bool'] = df['pred_prob'] > pred_thresh
        
        if by_patients:
            # take the first alarm incident for patients whose risk exceeded threshold
            first_alarm = df[df['pred_bool']].groupby('ikn').first()
            
            # take the last session for patients whose risk never exceeded threshold
            low_risk_ikns = set(df['ikn']).difference(first_alarm.index)
            last_session = df[df['ikn'].isin(low_risk_ikns)].groupby('ikn').last()
            
            matrix = compute_service_request_matrix(pd.concat([first_alarm, last_session]))
        else:
            matrix = compute_service_request_matrix(df)
            
        matrices.append(matrix)
    return pd.concat(matrices, keys=thresholds, names=['Threshold', 'Prediction']).T
    
def compute_service_request_matrix(df):
    result = pd.DataFrame()
    mapping = {False: 'Survives', True: 'Dies'}
    for pred, group in df.groupby('pred_bool'):
        counts = group['requested_service'].value_counts()
        result.loc[mapping[pred], 'Requested Service'] = counts.get(True, 0)
        result.loc[mapping[pred], 'Did Not Request Service'] = counts.get(False, 0)
    return result.astype(int)
