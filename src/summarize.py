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
from collections import defaultdict
from functools import partial

from sklearn.metrics import (
    average_precision_score, 
    precision_score, 
    recall_score, 
    roc_auc_score
)
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm
import numpy as np
import pandas as pd

from src.config import (
    cytopenia_gradings, 
    symptom_cols, 
    cancer_code_mapping,     
    subgroup_map, 
    group_title_map,
    variable_groupings_by_keyword
)
from src.utility import (
    compute_bootstrap_scores, 
    get_clean_variable_name, 
    get_clean_variable_names, 
    get_cyto_rates,
    get_units, 
    most_common_categories,
    nadir_bootstrap_worker,
    outcome_recall_score,
    split_and_parallelize, 
    twolevel, 
)

import logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', 
    datefmt='%I:%M:%S'
)

###############################################################################
# Data Characteristics
###############################################################################
class DataPartitionSummary:
    """Computes total number of patients and treatment sessions in different 
    population subgroups of a data partition
    """
    def __init__(self, X, Y, partition, top_categories=None):
        """
        Args:
            X (pd.DataFrame): table of the original data partition (not one-hot
                encoded, normalized, clipped, etc)
            Y (pd.DataFrame): target labels
            partition (str): name of the data partition e.g. All, Testing
            top_categories (dict): mapping of categorical columns (str) and 
                their most common categories (list of str)
        """
        self.top_categories = top_categories if top_categories else {}
        self.patient_idxs = X['ikn'].drop_duplicates(keep='last').index
        p = 'patients'
        s = 'sessions'
        self.total = {p: len(self.patient_idxs), s: len(X)}
        self.col = {
            p: (partition, f'Patients (N={self.total[p]})'),
            s: (partition, f'Treatment Sessions (N={self.total[s]})')
        }
        self.X = X
        self.Y = Y.loc[X.index]
        self.display = lambda x, total: f"{x} ({x/total*100:.1f})"
        
    def num_sessions_per_patient_summary(self, summary_df):
        num_sessions = self.X.groupby('ikn').apply(len)
        mean = num_sessions.mean()
        std = num_sessions.std()
        name = ('Number of Treatments, Mean (SD)', '')
        summary_df.loc[name, self.col['patients']] = f'{mean:.1f} ({std:.1f})'

    def median_age_summary(self, summary_df):
        for with_respect_to, col in self.col.items():
            if with_respect_to == 'patients':
                age = self.X.loc[self.patient_idxs, 'age']
            elif with_respect_to == 'sessions':
                age = self.X['age']
            q25, q75 = np.percentile(age, [25, 75]).astype(int)
            name = ('Age, Median (IQR)', '')
            summary_df.loc[name, col] = f"{int(age.median())} ({q25}-{q75})"

    def sex_summary(self, summary_df):
        mask = self.X['sex'] == 'F'
        self._summary(summary_df, mask, name=('Sex', 'Female, No. (%)'))
        self._summary(summary_df, ~mask, name=('Sex', 'Male, No. (%)'))
            
    def income_summary(self, summary_df):
        col = 'neighborhood_income_quintile'
        title = col.replace('_', ' ').title()
        for income_quintile in sorted(self.X[col].dropna().unique()):
            mask =  self.X[col] == income_quintile
            name = (title, f'Q{int(income_quintile)}, No. (%)')
            self._summary(summary_df, mask, name)
            
    def area_density_summary(self, summary_df):
        self._summary(
            summary_df, mask=self.X['rural'], 
            name=('Area of Residence', 'Rural, No. (%)')
        )
        self._summary(
            summary_df, mask=~self.X['rural'], 
            name=('Area of Residence', 'Urban, No. (%)')
        )
                
    def immigration_summary(self, summary_df):
        self._summary(
            summary_df, mask=self.X['is_immigrant'], 
            name=('Immigration', 'Immigrant, No. (%)')
        )
        self._summary(
            summary_df, mask=~self.X['is_immigrant'], 
            name=('Immigration', 'Non-Immigrant, No. (%)')
        )
        self._summary(
            summary_df, mask=self.X['speaks_english'], 
            name=('Immigration', 'English Speaker, No. (%)')
        )
        self._summary(
            summary_df, mask=~self.X['speaks_english'], 
            name=('Immigration', 'Non-English Speaker, No. (%)')
        )
        self._summary(
            summary_df, mask=self.X['years_since_immigration'] < 10,
            name=('Immigration', 'Arrival < 10 years, No. (%)')
        )
        self._summary(
            summary_df, mask=self.X['years_since_immigration'] >= 10,
            name=('Immigration', 'Arrival >= 10 years, No. (%)')
        )
        
    def world_region_of_birth_summary(self, summary_df):
        col = 'world_region_of_birth'
        for region in self.X[col].unique():
            if region == 'Unknown': continue
            mask = self.X[col] == region
            name = ('Immigrant World Region of Birth', f'{region}, No. (%)')
            self._summary(summary_df, mask, name)
            
    def category_subgroup_summary(self, summary_df, catcol='regimen', top=5):
        title = get_clean_variable_name(catcol)
        
        if catcol not in self.top_categories:
            top_cats = most_common_categories(self.X, catcol=catcol, top=top)
            top_cats = list(top_cats)
            self.top_categories[catcol] = top_cats
        else:
            top_cats = self.top_categories[catcol]
        
        for category in top_cats:
            mask = self.X[catcol] == category
            if catcol in ['curr_topog_cd', 'curr_morph_cd']: 
                category = cancer_code_mapping[category]
            name = (title, f'{category}, No. (%)')
            self._summary(summary_df, mask, name)
        
        # summarize the items not in top_items
        mask = ~self.X[catcol].isin(top_cats)
        self._summary(summary_df, mask, name=(title, 'Other'))
        
    def combordity_summary(self, summary_df):
        self._summary(
            summary_df, mask=self.X['diabetes'], 
            name=('Combordity', 'Diabetes, No. (%)')
        )
        self._summary(
            summary_df, mask=~self.X['diabetes'], 
            name=('Combordity', 'Non-Diabetes, No. (%)')
        )
        self._summary(
            summary_df, mask=self.X['hypertension'], 
            name=('Immigration', 'Hypertension, No. (%)')
        )
        self._summary(
            summary_df, mask=~self.X['hypertension'], 
            name=('Immigration', 'Non-Hypertension, No. (%)')
        )
    
    def gcsf_summary(self, summary_df):
        mask = self.X['age'] >= 65 & self.X['ODBGF_given']
        name = ('GCSF Administered, No. (%)', '')
        self._summary(summary_df, mask, name)
            
    def ckd_summary(self, summary_df):
        mask = self.X['baseline_eGFR'] < 60
        name = ('CKD prior to treatment, No. (%)', '')
        self._summary(summary_df, mask, name)
    
    def dialysis_summary(self, summary_df):
        mask = self.X['dialysis']
        name = ('Dialysis after treatment, No. (%)', '')
        self._summary(summary_df, mask, name)
            
    def target_summary(self, summary_df):
        for target, Y in self.Y.iteritems():
            name = ('Target Event', f'{target}, No. (%)')
            self._summary(summary_df, Y, name)
            
    def _summary(self, summary_df, mask, name):
        for with_respect_to, col in self.col.items():
            total = self.total[with_respect_to]
            if with_respect_to == 'patients':
                num_events = self.X.loc[mask, 'ikn'].nunique()
            elif with_respect_to == 'sessions':
                num_events = mask.sum()
            summary_df.loc[name, col] = self.display(num_events, total)
    
    def get_summary(
        self, 
        summary_df, 
        top=3, 
        include_target=True, 
        include_combordity=False,        
        include_gcsf=False, 
        include_ckd=False, 
        include_dialysis=False
    ):
        self.num_sessions_per_patient_summary(summary_df)
        self.median_age_summary(summary_df)
        self.immigration_summary(summary_df)
        self.world_region_of_birth_summary(summary_df)
        self.income_summary(summary_df)
        self.area_density_summary(summary_df)
        self.sex_summary(summary_df)
        self.category_subgroup_summary(summary_df, catcol='regimen', top=top)
        self.category_subgroup_summary(summary_df, catcol='curr_topog_cd', top=top)
        self.category_subgroup_summary(summary_df, catcol='curr_morph_cd', top=top)
        if include_target: self.target_summary(summary_df)
        if include_combordity: self.combordity_summary(summary_df)
        if include_gcsf: self.gcsf_summary(summary_df)
        if include_ckd: self.ckd_summary(summary_df)
        if include_dialysis: self.dialysis_summary(summary_df)

def data_characteristic_summary(
    eval_models, 
    save_dir, 
    partition='split', 
    target_event=None, 
    **kwargs
):
    """Get characteristics summary of patients and treatments for each 
    data split (Train-Valid-Test) or cohort (Development-Testing)
    
    Development cohort refers to Training and Validation split
    
    Args:
        partition (str): method to partition the data for summarization, either
            by split or cohort
        target_event (str): get summarization for the portion of data in which 
            target event has occured.
    """
    model_data = eval_models.orig_data
    labels = eval_models.labels
    summary_df = pd.DataFrame(index=twolevel, columns=twolevel)
    
    # Data full summary
    Y = pd.concat(labels.values())
    X = model_data.loc[Y.index]
    dps = DataPartitionSummary(X, Y, 'All')
    dps.get_summary(summary_df, **kwargs)
    top_categories = dps.top_categories
    
    # Data target event occured summary
    if target_event is not None:
        X = X[Y[target_event]]
        Y = Y.loc[X.index]
        dps = DataPartitionSummary(
            X, Y, target_event, top_categories=top_categories
        )
        dps.get_summary(summary_df, **kwargs)
    
    # Data partition summary
    if partition == 'split':
        # Train, Valid, Test data splits
        groupings = labels.items()
    elif partition == 'cohort':
        # Development, Testing cohort
        groupings = {
            'Development': pd.concat([labels['Train'], labels['Valid']]),
            'Testing': labels['Test']
        }
    for partition_name, Y in groupings.items():
        X = model_data.loc[Y.index]
        dps = DataPartitionSummary(
            X, Y, partition_name, top_categories=top_categories
        )
        dps.get_summary(summary_df, **kwargs)
        
    summary_df.to_csv(f'{save_dir}/data_characteristic_summary.csv')
    return summary_df

def top_cancer_regimen_summary(data, top=10):
    result = pd.DataFrame()
    top_cancers = most_common_categories(
        data, catcol='curr_topog_cd', top=top # cancer locations
    ) 
    for cancer_code, nsessions in top_cancers.items():
        cancer_name = cancer_code_mapping[cancer_code]
        mask = data['curr_topog_cd'] == cancer_code
        top_regimens = most_common_categories(
            data[mask], catcol='regimen', top=top
        )
        top_regimens = [f'{k} ({v})' for k, v in top_regimens.items()]
        result[f'{cancer_name} (N={nsessions})'] = top_regimens
    result.index += 1
    return result

###############################################################################
# Input Features
###############################################################################
def feature_summary(
    eval_models, 
    prep, 
    target_keyword, 
    save_dir, 
    deny_old_survey=True
):
    """
    Args:
        deny_old_survey (bool): If True, considers survey (symptom 
            questionnaire) response more than 5 days prior to chemo session as
            "missing" data
    """
    df = prep.dummify_data(eval_models.orig_data.copy())
    train_idxs = eval_models.labels['Train'].index
    df = df.loc[train_idxs]
    N = len(df)

    # remove missingness features, targets, first visit date, and ikn
    cols = df.columns
    mask = cols.str.contains('is_missing') | cols.str.contains(target_keyword)
    df = df[cols[~mask].drop('ikn')]

    # get number of missing values, mean, and standard deviation for each 
    # feature for the training set
    summary = df.astype(float).describe()
    summary = summary.loc[['count', 'mean', 'std']].T
    summary = summary.round(6)
    summary['count'] = len(train_idxs) - summary['count']
    
    if deny_old_survey:
        event_dates = prep.event_dates.loc[train_idxs]
        for col in symptom_cols:
            if col not in summary.index:
                continue
            survey_date = event_dates[f'{col}_survey_date']
            days_before_chemo = (event_dates['visit_date'] - survey_date).dt.days
            summary.loc[col, 'count'] = sum(days_before_chemo > 5)
    
    summary['Missingness (%)'] = (summary['count'] / N * 100).round(1)
    
    format_arr = lambda arr: arr.round(2).astype(str)
    mean, std = format_arr(summary['mean']), format_arr(summary['std'])
    summary['Mean (SD)'] = mean + ' (' + std + ')'
    
    name_map = {
        'count': f'Train (N={N}) - Missingness Count', 
        'mean': 'Train - Mean', 
        'std': 'Train - SD'
    }
    summary = summary.rename(columns=name_map)

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
def nadir_summary(
    df, 
    output_path, 
    cytopenia='Neutropenia', 
    load_ci=False, 
    n_bootstraps=1000, 
    processes=32
):
    """
    Args:
        load_ci (bool): If True, loads the bootstrapped nadir days to compute 
            confidence interval
    """
    if cytopenia not in {'Neutropenia', 'Anemia', 'Thrombocytopenia'}: 
        raise ValueError('cytopenia must be one of Neutropneia, Anemia, or '
                         'Thrombocytopenia')
        
    if load_ci:
        filename = f'nadir_{cytopenia}_bootstraps'
        ci_df = pd.read_csv(f'{output_path}/data/analysis/{filename}.csv')
        ci_df = ci_df.set_index('regimen')
    else:
        ci_df = pd.DataFrame()
        
    cycle_lengths = dict(df[['regimen', 'cycle_length']].values)
    result = {}
    for regimen, group in tqdm(df.groupby('regimen')):
        cycle_length = int(cycle_lengths[regimen])
        days = range(0, cycle_length)
        result[regimen] = {
            'NSessions': len(group), 
            'Cycle Length': cycle_length
        }
        
        for grade, thresholds in cytopenia_gradings.items():
            if cytopenia not in thresholds: 
                continue
                
            thresh = thresholds[cytopenia]   
            cyto_rates_per_day = get_cyto_rates(group[days], thresh)
            name = f'{grade} {cytopenia} Rate (<{thresh})'
            
            if all(cyto_rates_per_day == 0):
                # if no cytopenia was observed for all days
                result[regimen][name] = '0 (0-1)'
                continue
                
            nadir_day = np.argmax(cyto_rates_per_day)
            nadir_day_measurements = group[nadir_day].dropna()
            nadir_day_n_events = (nadir_day_measurements < thresh).sum()
            if nadir_day_n_events < 5: 
                # can't allow small cells less than 5 according to ICES 
                # privacy policy
                result[regimen][name] = '0 (0-1)'
                continue

            worst_cyto_rate = cyto_rates_per_day[nadir_day].round(3)

            # binomial confidence interval for cytopenia rate (since we are
            # working with binomial distribution, i.e. cytopenia - 1, not 
            # cytopenia - 0)
            ci_lower, ci_upper = proportion_confint(
                count=nadir_day_n_events,nobs=len(nadir_day_measurements), 
                method='wilson', alpha=(1-0.95) # 95% CI
            )
            ci_lower, ci_upper = ci_lower.round(3), ci_upper.round(3)

            if grade == 'Grade 2':
                # get 95% confidence interval for nadir day using bootstrap 
                # technique
                if regimen not in ci_df.index:
                    bootstraps = range(n_bootstraps)
                    worker = partial(
                        nadir_bootstrap_worker, df=group, days=days, 
                        thresh=thresh
                    )
                    ci_df.loc[regimen, bootstraps] = split_and_parallelize(
                        bootstraps, worker, split_by_ikns=False, 
                        processes=processes
                    )
                nadir_days = ci_df.loc[regimen].values
                lower, upper = np.percentile(nadir_days, [2.5, 97.5]).astype(int)
                # set day 1 as day of administration (not day 0)
                result[regimen]['Nadir Day'] = f'{nadir_day+1} ({lower}-{upper})'
                
                # col = 'NMeasurements at Nadir Day'
                # result[regimen][col] = len(nadir_day_measurements)
                
            result[regimen][name] = f'{worst_cyto_rate} ({ci_lower}-{ci_upper})'
        
    summary_df = pd.DataFrame(result).T
    
    # write the results
    summary_df.to_csv(
        f'{output_path}/data/analysis/nadir_{cytopenia}_summary.csv'
    )
    ci_df.to_csv(
        f'{output_path}/data/analysis/nadir_{cytopenia}_bootstraps.csv', 
        index_label='regimen'
    )
    
    return summary_df

###############################################################################
# Subgroup Performance
###############################################################################
class SubgroupPerformanceSummary:
    def __init__(
        self, 
        algorithm, 
        eval_models, 
        target_events,      
        pred_thresh=0.2, 
        split='Test', 
        display_ci=False, 
        load_ci=False,
        digit_round=3,
        include_outcome_recall=False,
        **kwargs
    ):
        """
        Args:
            pred_thresh (float or list): prediction threshold (float) for alarm 
                trigger (when model outputs a risk probability greater than 
                prediction threshold, alarm is triggered) or sequence of 
                prediction thresholds corresponding to each target event
            include_outcome_recall (bool): include outcome-level recall scores.
                Only ED/H/D events are supported.
            **kwargs: keyword arguments fed into outcome_recall_score (if we 
                are including outcome-level recall)
        """
        self.algorithm = algorithm
        self.split = split
        self.target_events = target_events
        self.display_ci = display_ci
        self.pred_thresh = pred_thresh
        
        # used for getting most common category subgroups (of a categorical 
        # column)
        self.entire_data = eval_models.orig_data
        
        self.Y = eval_models.labels[split]
        self.data = self.entire_data.loc[self.Y.index]
        self.N = self.data['ikn'].nunique()
        self.preds = eval_models.preds[split][algorithm]

        self.ci_df = pd.DataFrame(index=twolevel)
        if load_ci:
            filename = f'bootstrapped_subgroup_scores_{algorithm}.csv'
            self.ci_df = pd.read_csv(
                f'{eval_models.output_path}/confidence_interval/{filename}', 
                index_col=[0,1]
            )
            self.ci_df.columns = self.ci_df.columns.astype(int)
            
        self.round = lambda x: np.round(x, digit_round)
        
        self.include_outcome_recall = include_outcome_recall
        self.kwargs = kwargs

    def entire_test_set_score(self, summary_df):
        name = (f'Entire {self.split} Cohort', f'{self.N} patients')
        mask = self.data['ikn'].notnull()
        self._subgroup_score(summary_df, mask, name)
    
    def age_subgroups_score(self, summary_df):
        for (low, high) in [(18, 64), (65, np.inf)]:
            mask = self.data['age'].between(low, high)
            interval = f'{low}+' if high == np.inf else f'{low}-{high}'
            name = ('Age',  interval)
            self._subgroup_score(summary_df, mask, name)
            
    def sex_subgroups_score(self, summary_df):
        mask = self.data['sex'] == 'F'
        self._subgroup_score(summary_df, mask, ('Sex', 'Female'))
        self._subgroup_score(summary_df, ~mask, ('Sex', 'Male'))
            
    def immigrant_subgroups_score(self, summary_df):
        self._subgroup_score(
            summary_df, mask=self.data['is_immigrant'], 
            name=('Immigration', 'Immigrant')
        )
        self._subgroup_score(
            summary_df, mask=~self.data['is_immigrant'], 
            name=('Immigration', 'Non-Immigrant')
        )
        self._subgroup_score(
            summary_df, mask=self.data['speaks_english'], 
            name=('Immigration', 'English Speaker')
        )
        self._subgroup_score(
            summary_df, mask=~self.data['speaks_english'], 
            name=('Immigration', 'Non-English Speaker')
        )
        self._subgroup_score(
            summary_df, mask=self.data['years_since_immigration'] < 10,
            name=('Immigration', 'Arrival < 10 years')
        )
        self._subgroup_score(
            summary_df, mask=self.data['years_since_immigration'] >= 10, 
            name=('Immigration', 'Arrival >= 10 years')
        )
            
    def world_region_of_birth_subgroups_score(self, summary_df):
        col = 'world_region_of_birth'
        for region in self.data[col].unique():
            if region == 'Unknown' or region == 'Other': continue
            mask = self.data[col] == region
            name = ('Immigrant World Region of Birth', region)
            self._subgroup_score(summary_df, mask, name)
            
    def income_subgroups_score(self, summary_df):
        col = 'neighborhood_income_quintile'
        for income_quintile in self.data[col].dropna().unique():
            mask = self.data[col] == income_quintile
            name = ('Neighborhood Income Quintile', f'Q{int(income_quintile)}')
            self._subgroup_score(summary_df, mask, name)
            
    def area_density_subgroups_score(self, summary_df):
        self._subgroup_score(
            summary_df, mask=self.data['rural'], 
            name=('Area of Residence', 'Rural')
        )
        self._subgroup_score(
            summary_df, mask=~self.data['rural'], 
            name=('Area of Residence', 'Urban')
        )
            
    def most_common_category_subgroups_score(
        self, 
        summary_df, 
        catcol, 
        mapping=None,
        top=3,
    ):
        title = get_clean_variable_name(catcol)
        top_cats = most_common_categories(
            self.entire_data, catcol=catcol, top=top
        )
        top_cats = list(top_cats)
        for category in top_cats:
            mask = self.data[catcol] == category
            if mapping is not None: category = mapping[category]
            name = (title, category)
            self._subgroup_score(summary_df, mask, name)

    def days_since_starting_subgroups_score(self, summary_df):
        for (low, high) in [(0, 30), (31, 90), (91, np.inf)]:
            mask = self.data['days_since_starting_chemo'].between(low, high)
            interval = f'{low}+' if high == np.inf else f'{low}-{high}'
            name = ('Days Since Starting Regimen', interval)
            self._subgroup_score(summary_df, mask, name)
    
    def cycle_length_subgroups_score(self, summary_df):
        for cycle_length in self.data['cycle_length'].unique():
            mask = self.data['cycle_length'] == cycle_length
            name = ('Cycle Length',  str(cycle_length))
            self._subgroup_score(summary_df, mask, name)
            
    def ckd_subgroups_score(self, summary_df):
        mask = self.data['baseline_eGFR'] < 60
        title = 'CKD Prior to Treatment'
        self.score_within_subgroups(summary_df, mask, (title, 'Y'))
        self.score_within_subgroups(summary_df, ~mask, (title, 'N'))
            
    def get_bootstrapped_scores(
        self, 
        Y_true, 
        Y_pred, 
        name, 
        target_event, 
        n_bootstraps=10000
    ):
        group_name, subgroup_name = name
        group_name = group_name.lower().replace(' ', '_')
        # WARNING: SUPER HARD CODED
        subgroup_name = subgroup_name.split('(')[0].strip().lower()
        subgroup_name = subgroup_name.replace(' ', '_').replace('/', '_')
        
        ci_index = f'{group_name}_{subgroup_name}_{target_event}'
        if ci_index not in self.ci_df.index:
            auc_scores = compute_bootstrap_scores(
                Y_true.reset_index(drop=True), Y_pred, n_bootstraps=n_bootstraps
            )
            logging.info(f'Completed bootstrap computations for {ci_index}')
            aurocs, auprcs = np.array(auc_scores).T
            self.ci_df.loc[(ci_index, 'AUROC'), range(n_bootstraps)] = aurocs
            self.ci_df.loc[(ci_index, 'AUPRC'), range(n_bootstraps)] = auprcs
        return self.ci_df.loc[ci_index].values
        
    def get_confidence_interval(
        self, 
        Y_true, 
        Y_pred, 
        summary_df, 
        name, 
        target_event
    ):
        aurocs, auprcs = self.get_bootstrapped_scores(
            Y_true, Y_pred, name, target_event
        )
        for metric, scores in [('AUROC', aurocs), ('AUPRC', auprcs)]:
            col = (target_event, metric)
            lower, upper = np.percentile(scores, [2.5, 97.5]).round(3)
            display = f'{summary_df.loc[name, col]} ({lower}-{upper})'
            summary_df.loc[name, col] = display
        return summary_df

    def _subgroup_score(self, summary_df, mask, name):
        # extract the labels and preds for the subgroup
        Y = self.Y[mask]
        pred_prob = self.preds.loc[Y.index]
        
        # add the proportion of patients in the subgroup to the name
        subgroup, category = name
        num_patients = self.data.loc[mask, 'ikn'].nunique()
        name = (subgroup, f'{category} ({num_patients/self.N*100:.1f}%)')
        
        # compute the scores
        pred_thresh = self.pred_thresh
        desc = f'Computing scores for {category} {subgroup}...'
        for idx, target_event in enumerate(tqdm(self.target_events, desc=desc)):
            Y_true = Y[target_event]
            Y_pred_prob = pred_prob[target_event]
            if isinstance(pred_thresh, list): pred_thresh = pred_thresh[idx]
                
            Y_pred_bool = Y_pred_prob > pred_thresh
            if Y_true.nunique() < 2:
                logging.warning('No positive examples, skipping '
                                f'{target_event} - {name}')
                continue
            
            # AUROC/AUPRC
            auroc = roc_auc_score(Y_true, Y_pred_prob)
            auprc = average_precision_score(Y_true, Y_pred_prob)
            summary_df.loc[name, (target_event, 'AUROC')] = self.round(auroc)
            summary_df.loc[name, (target_event, 'AUPRC')] = self.round(auprc)
            if self.display_ci:
                summary_df = self.get_confidence_interval(
                    Y_true, Y_pred_prob, summary_df, name, target_event
                )
                
            # PPV/Recall
            ppv = precision_score(Y_true, Y_pred_bool, zero_division=1)
            recall = recall_score(Y_true, Y_pred_bool, zero_division=1)
            summary_df.loc[name, (target_event, 'PPV')] = self.round(ppv)
            summary_df.loc[name, (target_event, 'Recall')] = self.round(recall)
            if self.include_outcome_recall:
                outcome_recall = outcome_recall_score(
                    Y_true, Y_pred_bool, target_event, **self.kwargs
                )
                col = (target_event, 'Outcome-Level Sensitivity')
                summary_df.loc[name, col] = self.round(outcome_recall)
            
            # Warning Rate/Event Rate
            # warning_rate = self.round(Y_pred_bool.mean())
            # summary_df.loc[name, (target_event, 'Warning Rate')] = warning_rate
            event_rate = self.round(Y_true.mean())
            summary_df.loc[name, (target_event, 'Event Rate')] = event_rate
            
    def get_summary(self, subgroups, summary_df):
        if 'all' in subgroups: self.entire_test_set_score(summary_df)
        if 'age' in subgroups: self.age_subgroups_score(summary_df)
        if 'sex' in subgroups: self.sex_subgroups_score(summary_df)
        if 'immigrant' in subgroups: self.immigrant_subgroups_score(summary_df)
        if 'world_region_of_birth' in subgroups: 
            self.world_region_of_birth_subgroups_score(summary_df)
        if 'income' in subgroups: self.income_subgroups_score(summary_df)
        if 'area_density' in subgroups: 
            self.area_density_subgroups_score(summary_df)
        if 'regimen' in subgroups: 
            self.most_common_category_subgroups_score(summary_df, 'regimen')
        if 'cancer_location' in subgroups: 
            self.most_common_category_subgroups_score(
                summary_df, 'curr_topog_cd', mapping=cancer_code_mapping
            )
        if 'days_since_starting' in subgroups: 
            self.days_since_starting_subgroups_score(summary_df)
        if 'cycle_length' in subgroups: 
            self.cycle_length_subgroups_score(summary_df)
        if 'ckd' in subgroups: self.ckd_subgroups_score(summary_df)
            
def subgroup_performance_summary(
    algorithm, 
    eval_models, 
    pred_thresh=0.2, 
    subgroups=None, 
    target_events=None,                          
    display_ci=False, 
    load_ci=False, 
    save_ci=False,
    include_outcome_recall=False,
    **kwargs
):
    """
    Args:
        display_ci (bool): If True, display confidence interval
        load_ci (bool): If True, load saved bootstrapped scores for computing 
            confidence interval
        pred_thresh (float or list): prediction threshold (float) for alarm 
            trigger (when model outputs a risk probability greater than 
            prediction threshold, alarm is triggered) or sequence of prediction
            thresholds corresponding to each target event
        include_outcome_recall (bool): include outcome-level recall scores.
            Only ED/H/D events are supported.
        **kwargs: keyword arguments fed into outcome_recall_score (if we 
            are including outcome-level recall)
    """
    if target_events is None:
        target_events = eval_models.target_events
        
    if subgroups is None:
        subgroups = {
            'all', 'age', 'sex', 'immigrant', 'world_region_of_birth', 'income', 
            'area_density', 'regimen', 'cancer_location', 'days_since_starting'
        }
        
    summary_df = pd.DataFrame(index=twolevel, columns=twolevel)
    sps = SubgroupPerformanceSummary(
        algorithm, eval_models, target_events, pred_thresh=pred_thresh, 
        display_ci=display_ci, load_ci=load_ci, 
        include_outcome_recall=include_outcome_recall, **kwargs
    )
    sps.get_summary(subgroups, summary_df)
    
    # write results
    save_dir = eval_models.output_path
    if save_ci: 
        filename = f'bootstrapped_subgroup_scores_{algorithm}'
        sps.ci_df.to_csv(f'{save_dir}/confidence_interval/{filename}.csv')
    filename = f'subgroup_performance_summary_{algorithm}'
    summary_df.to_csv(f'{save_dir}/tables/{filename}.csv')
    
    return summary_df

def get_worst_performing_subgroup(
    eval_models, 
    catcol='regimen', 
    algorithm='XGB', 
    target_event='Neutropenia', 
    split='Valid'
):
    """Analyze subgroups with the worst performance (usually in the validation
    set)
    """
    summary_df = pd.DataFrame(columns=twolevel)
    sps = SubgroupPerformanceSummary(
        algorithm, eval_models, target_events=[target_event], split=split
    )
    for regimen, group in sps.data.groupby(catcol):
        Y_subgroup = sps.Y.loc[group.index]
        sps.score_within_subgroups(Y_subgroup, regimen, summary_df)
    summary_df[(target_event, 'NSessions')] = sps.data[catcol].value_counts()
    summary_df = summary_df.sort_values(by=(target_event, 'AUROC'))
    summary_df.index = [cancer_code_mapping.get(index, index) 
                        for index in summary_df.index]
    return summary_df

###############################################################################
# Palliative Care Consultation Service (PCCS) (TODO: Move to Utility)
###############################################################################
def pccs_receival_summary(eval_models, event_dates, split='Test', **kwargs):
    """Determine number of patients that received early palliative care 
    consultation service (PCCS).
    
    We take the first session in which model triggered an alarm 
    (risk prediction > threshold), and determine if palliative consultation 
    service was received within the appropriate time frame. If model never 
    triggered an alarm, we use the last session of that patient.
    
    We then group the numbers by the predicted or observed outcome to create a 
    2 by 2 table (Will/will not experience target event vs Did/did not request 
    service)
    
    We create this 2 by 2 table for multiple subgroups (immigrants, sex, 
    quintiles, etc)
    
    Args:
        **kwargs: keyword arguments fed into get_pccs_analysis_data
            
    Returns:
        A nested dictionary {subgroup_type: {outcome_type: 2x2 pd.DataFrame}}
    """    
    df = get_pccs_analysis_data(
        eval_models, event_dates, split=split, **kwargs
    )
    
    result = defaultdict(dict)
    for outcome_type in ['predicted', 'observed']:
        
        # Get summary for entire cohort 
        matrix = pccs_receival_matrix(df, group_by=outcome_type)
        matrix = matrix.rename_axis(outcome_type.title()).T
        result[outcome_type][f'Entire {split} Cohort'] = matrix
        
        # Get summary for each subgroup populations
        for col, mapping in subgroup_map.items():
            matrices = {}
            for subgroup_name, group, in df.groupby(col):
                name = mapping.get(subgroup_name, subgroup_name)
                skip_subgroups = mapping.get('do_not_include', [])
                if subgroup_name in skip_subgroups: continue
                matrices[name] = pccs_receival_matrix(
                    group, group_by=outcome_type
                ) 
            
            title = group_title_map[col]
            names = [title, outcome_type.title()]
            matrices = pd.concat(matrices, names=names).T
            result[outcome_type][title] = matrices

    return result

def get_pccs_analysis_data(
    eval_models, 
    event_dates,
    time_frame=(5*365,90), 
    pred_thresh=0.5,             
    algorithm='ENS', 
    split='Test', 
    target_event='365d Mortality',
    verbose=True,
):
    """
    Args:
        time_frame (tuple(int, int)): A tuple of integers representing the 
            appropriate time frame within which PCCS was received, in terms of 
            number of days before and after a treatment session
    """
    if verbose:
        logging.info('Arranging PCCS Analysis Data...')
    df = pd.DataFrame()
    
    # Get prediction (risk) and observation of target event
    df['predicted_prob'] = eval_models.preds[split][algorithm][target_event]
    df['predicted'] = df['predicted_prob'] > pred_thresh
    df['observed'] = eval_models.labels[split][target_event]
    
    # Get patient id, immigrant status, sex, income quintile, etc
    cols = list(subgroup_map)+['ikn']
    df[cols] = eval_models.orig_data.loc[df.index, cols]
    # need to binarize time since immigration arrival
    df['years_since_immigration'] = df['years_since_immigration'] < 10 
    
    # Determine if patient received PCCS within N days before to M days after 
    # treatment
    df['received_pccs'] = check_received_pccs(
        event_dates.loc[df.index], time_frame
    )
    
    # Keep original index
    df = df.reset_index()
    
    # Take the first session in which alarm was triggered for each patient
    first_alarm = df[df['predicted']].groupby('ikn').first()
    # or the last session if no alarms were ever triggered
    # (patients whose risk never exceeded threshold)
    low_risk_ikns = set(df['ikn']).difference(first_alarm.index)
    last_session = df[df['ikn'].isin(low_risk_ikns)].groupby('ikn').last()
    df = pd.concat([first_alarm, last_session])
    if verbose:
        logging.info('Number of patients with first alarm incidents '
                     f'(risk > {pred_thresh:.2f}): {len(first_alarm)}. Number '
                     'of patients with no alarm incidents (take the last '
                     f'session): {len(last_session)}')
    
    return df
    
def pccs_receival_matrix(df, group_by):
    result = pd.DataFrame()
    mapping = {False: 'Survives', True: 'Dies'}
    col = 'Received PCCS'
    for status, group in df.groupby(group_by):
        status = mapping[status]
        counts = group['received_pccs'].value_counts()
        result.loc[status, col] = counts.get(True, 0)
        result.loc[status, f'Not {col}'] = counts.get(False, 0)
    result[f'{col} (%)'] = result[col] / result.sum(axis=1) * 100
    return result

def check_received_pccs(event_dates, time_frame):
    days_before, days_after = time_frame
    service_date = event_dates['PCCS_date']
    visit_date = event_dates['visit_date']
    return service_date.between(visit_date - pd.Timedelta(days=days_before),
                                visit_date + pd.Timedelta(days=days_after))

###############################################################################
# Chemotherapy Near End-of-Life (TODO: Move to Utility)
###############################################################################
def eol_chemo_receival_summary(eval_models, event_dates, split='Test', **kwargs):
    """Determine number of patients that received chemotherapy near end-of-life
    (EOL), for multiple subgroups (immigrants, sex, quintiles, etc)
    
    We take the first session in which model triggered an alarm 
    (risk prediction > threshold), and determine if patient died within X days 
    (experienced target event). If model never triggered an alarm, we use the 
    last session of that patient.
    
    Args:
        **kwargs: keyword arguments fed into get_eol_chemo_analysis_data
    
    Returns:
        A Table (pd.DataFrame) of number and proportion of patients that 
        recieved chemotherapy at EOL among the different subgroup populations
    """
    df = get_eol_chemo_analysis_data(
        eval_models, event_dates, split=split, **kwargs
    )
        
    # Get summary for each subgroup populations
    summary = pd.DataFrame(columns=twolevel)
    eol_chemo_receival_matrix(df, (f'Entire {split} Cohort', ''), summary)
    for col, mapping in subgroup_map.items():
        title = group_title_map[col]
        for subgroup_name, group, in df.groupby(col):
            if subgroup_name in mapping.get('do_not_include', []): continue
            subgroup_name = mapping.get(subgroup_name, subgroup_name)
            eol_chemo_receival_matrix(group, (title, subgroup_name), summary)
        
    return summary

def get_eol_chemo_analysis_data(
    eval_models, 
    event_dates,
    pred_thresh=0.2,             
    algorithm='ENS', 
    split='Test', 
    target_event='30d Mortality',
    verbose=True,
):
    if verbose:
        logging.info('Arranging Chemo at End-of-Life Analysis Data...')
    df = pd.DataFrame()

    # Get prediction (risk) and observation of target event
    df['predicted_prob'] = eval_models.preds[split][algorithm][target_event]
    df['predicted'] = df['predicted_prob'] > pred_thresh
    df['observed'] = eval_models.labels[split][target_event]

    # Get patient id, immigrant status, sex, income quintile, etc
    cols = list(subgroup_map)+['ikn']
    df[cols] = eval_models.orig_data.loc[df.index, cols]
    # need to binarize time since immigration arrival
    df['years_since_immigration'] = df['years_since_immigration'] < 10 

    # Get death date and visit date
    cols = ['D_date', 'visit_date']
    df[cols] = event_dates.loc[df.index, cols]

    # Only keep patients that died
    mask = df['D_date'].notnull()
    if verbose:
        logging.info(f"Removing {df.loc[~mask, 'ikn'].nunique()} patients that "
                     f"did not die out of {df['ikn'].nunique()} total patients")
    df = df[mask]
    
    # Determine if patient received chemo near end of life (EOL)
    days_thresh = target_event.split(' ')[0]
    df['received_chemo_near_EOL'] = df['D_date'] - df['visit_date'] < days_thresh
    
    # Keep original index
    df = df.reset_index()
    
    # Take the first session in which alarm was triggered for each patient
    first_alarm = df[df['predicted']].groupby('ikn').first()
    # or the last session if no alarms were ever triggered
    # (patients whose risk never exceeded threshold)
    low_risk_ikns = set(df['ikn']).difference(first_alarm.index)
    last_session = df[df['ikn'].isin(low_risk_ikns)].groupby('ikn').last()
    df = pd.concat([first_alarm, last_session])
    if verbose:
        logging.info('Number of patients with first alarm incidents '
                     f'(risk > {pred_thresh:.2f}): {len(first_alarm)}. Number '
                     'of patients with no alarm incidents (take the last '
                     f'session): {len(last_session)}')
    
    return df
    
def eol_chemo_receival_matrix(group, col, summary):
    name = 'Received Chemo Near EOL'
    counts = group['received_chemo_near_EOL'].value_counts()
    summary.loc[name, col] = counts.get(True, 0)
    summary.loc[f'Not {name}', col] = counts.get(False, 0)
    total = summary[col].sum()
    summary.loc[f'{name} (%)', col] = summary.loc[name, col] / total * 100
