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
Module for intervention analysis (analyzing system's clinical utility impact)
"""
from functools import partial

import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from src import logging
from src.config import cancer_code_mapping
from src.summarize import SubgroupSummary
from src.utility import (
    get_first_alarms_or_last_treatment,
    twolevel
)

def get_intervention_analysis_data(
    eval_models, 
    event_dates,
    pred_thresh=0.5,             
    alg='ENS', 
    split='Test', 
    target_event='365d Mortality',
    verbose=True,
):
    """Extract the relevant data for intervention analysis.
    We want to analyze the first alarm incidents.
    """
    # Get prediction (risk) of target event
    pred_prob = eval_models.preds[split][alg][target_event]
    pred_prob = pred_prob.rename('predicted_prob').to_frame()
    
    # Get patient death date, first PCCS date, and treatment date
    df = pred_prob.join(event_dates)
    
    # Get patient id, immigrant status, sex, income quintile, etc
    cols = [
        'ikn', 'age', 'sex', 'neighborhood_income_quintile', 'rural', 
        'recent_immigrant', 'speaks_english', 'years_since_immigration',
        'world_region_of_birth', 'regimen', 'cancer_topog_cd', 
        'days_since_starting_chemo'
    ]
    df = df.join(eval_models.orig_data[cols])
    
    # Keep original index
    df = df.reset_index()
    
    # Take the first session in which an alarm was triggered for each patient
    # or the last session if alarms were never triggered
    df = get_first_alarms_or_last_treatment(df, pred_thresh, verbose=verbose)
    
    df = df.reset_index().set_index('index')
    return df

###############################################################################
# Palliative Care Consultation Service (PCCS)
###############################################################################
def get_pccs_analysis_data(
    eval_models, 
    event_dates,
    days_before_death=180,
    verbose=True,
    **kwargs
):
    """Extract the relevant data for analysis of palliative care consultation 
    service (PCCS) receival
    
    Args:
        days_before_death (int): Minimum number of days before death in which
            PCCS receival is still acceptable
    """
    if verbose:
        logging.info('Arranging PCCS Analysis Data...')
    
    df = get_intervention_analysis_data(
        eval_models, event_dates, verbose=verbose, **kwargs
    )
    
    # Get patient's vital status
    df['status'] = df['death_date'].isnull().replace({True: 'Alive', False: 'Dead'})

    # Determine if patient received early PCCS
    latest_date = df['death_date'] - pd.Timedelta(days=days_before_death)
    df['received_early_pccs'] = df['first_PCCS_date'] < latest_date
    # Determine if alert would result in patient receiving early PCCS
    # (i.e. alert was not too late)
    df['early_pccs_by_alert'] = df['predicted'] & (df['visit_date'] < latest_date)
    
    # Determine if patient's first visit date was within the minimum number of
    # days before death (meaning model could never have given an alert in time 
    # for patient to receive early PCCS)
    df['first_visit_near_death'] = df['first_visit_date'] > latest_date
    
    return df

def get_pccs_impact(df, no_alarm_strategy='no_pccs'):
    """Determine the model's clinical impact on allocation of palliative care 
    consulation service (PCCS). 

    We compute the number of patients who received PCCS for Usual Care and 
    System-Guided Care.

    System-Guided Care:
    Warning system triggers an alarm => patient receives PCCS
    Warning system does not trigger an alarm =>
        a) Strategy 1: patient does not receive PCCS
        b) Strategy 2: default to usual care

    Args:
        df (pd.DataFrame): table of patients and their relevant data (PCCS 
            receival, vital status, etc) (from output of get_pccs_analysis_data)
        no_alarm_strategy (str): which strategy to use in the absence of an
            alarm. Either 'no_pccs' (patient will not receive pccs) or 'uc'
            (default to whatever action occured in usual care)
    """
    usual_care = _get_pccs_impact(df, care_name='usual')
    system_care = _get_pccs_impact(df, care_name='system', no_alarm_strategy=no_alarm_strategy)
    impact = pd.DataFrame({'Usual Care': usual_care, 'System-Guided Care': system_care})
    return impact.astype(int)

def _get_pccs_impact(df, **kwargs):
    counts = get_pccs_receival(df, **kwargs)
    impact = {
        'Alive With PCCS': counts.loc['Got PCCS', 'Alive'], 
        'Died With Late PCCS': counts.loc['Got Late PCCS', 'Dead'], 
        'Died With Early PCCS': counts.loc['Got Early PCCS', 'Dead']
    }
    n_died = sum(df['status'] == 'Dead')
    n_pccs = sum(impact.values())
    impact['Died Without Early PCCS'] = n_died - impact['Died With Early PCCS']
    impact['Total With PCCS'] = n_pccs
    return impact

def get_pccs_receival_by_subgroup(df, subgroups=None, **kwargs):
    """Determine number of patients who received palliative care 
    consultation service (PCCS) for different subgroups (immigrants, sex, 
    income quintiles, etc).
    
    Args:
        df (pd.DataFrame): table of patients and their relevant data (subgroup 
            status, PCCS date, etc) (from output of get_pccs_analysis_data)
        **kwargs: keyword arguments fed into get_pccs_receival
    """   
    impact_func = partial(get_pccs_receival, **kwargs)
    si = SubgroupImpact(df, impact_func, subgroups=subgroups)
    summary = si.get_summary()
    return summary

def get_pccs_receival(df, care_name='usual', no_alarm_strategy='no_pccs'):
    """Determine number of patients who received palliative care 
    consultation service (PCCS).
    
    Args:
        df (pd.DataFrame): table of patients and their relevant data (visit 
            date, PCCS date, etc) (from output of get_pccs_analysis_data)
        care_name (str): Type of care used for allocating PCCS. Either 'system'
            for System-Guided Care or 'usual' for Usual Care
        no_alarm_strategy (str): which strategy to use in the absence of an
            alarm for System-Guided Care. Either 'no_pccs' (patient will not 
            receive pccs) or 'uc' (default to whatever action occured in usual 
            care)
    """
    if care_name == 'system': care_name = f'{care_name}-{no_alarm_strategy}'
    get_receival_mask = {
        'usual': lambda x: x['first_PCCS_date'].notnull(),
        'system-no_pccs': lambda x: x['predicted'],
        'system-uc': lambda x: x['predicted'] | x['first_PCCS_date'].notnull()
    }
    get_early_receival_mask = {
        'usual': lambda x: x['received_early_pccs'],
        'system-no_pccs': lambda x: x['early_pccs_by_alert'],
        'system-uc': lambda x: x['early_pccs_by_alert'] | x['received_early_pccs']
    }
    
    counts = {}
    for status, group in df.groupby('status'):
        result = {}
        
        receival_mask = get_receival_mask[care_name](group)
        result['Got PCCS'] = sum(receival_mask)
        result['Not Got PCCS'] = sum(~receival_mask)
        
        if status == 'Dead':
            early_receival_mask = get_early_receival_mask[care_name](group)
            result['Got Early PCCS'] = sum(early_receival_mask)
            result['Got Late PCCS'] = sum(receival_mask & ~early_receival_mask)
            
            col = 'Got Early PCCS (%)'
            result[col] = sum(early_receival_mask) / len(df) * 100
            result[f'Not {col}'] = sum(~early_receival_mask) / len(df) * 100
            
        counts[status] = result
    counts = pd.DataFrame(counts).fillna(0)
    return counts

###############################################################################
# Treatment Near End-of-Life
###############################################################################
def get_eol_treatment_analysis_data(
    eval_models, 
    event_dates,
    verbose=True,
    target_event='30d Mortality',
    **kwargs
):
    if verbose:
        logging.info('Arranging Treatment at End-of-Life Analysis Data...')
    
    df = get_intervention_analysis_data(
        eval_models, event_dates, verbose=verbose, target_event=target_event, 
        **kwargs
    )

    # Only keep patients that died
    mask = df['death_date'].isnull()
    if verbose:
        logging.info(f"Removing {sum(mask)} patients that did not die out of "
                     f"{len(mask)} total patients")
    df = df[~mask]
    
    # Determine if patient received chemo near end of life
    days_thresh = target_event.split(' ')[0]
    days_to_death = df['death_date'] - df['visit_date']
    df['received_treatment_near_EOL'] = days_to_death < days_thresh
    
    return df

def get_eol_treatment_receival_by_subgroup(df, subgroups=None):
    """Determine number of patients who received treatment near end-of-life
    (EOL) for multiple subgroups (immigrants, sex, income quintiles, etc)
    
    Args:
        df (pd.DataFrame): table of patients and their relevant data (subgroup 
            status, treatment receival, etc) (from output of 
            get_eol_treatment_analysis_data)
    
    Returns:
        A Table (pd.DataFrame) of number and proportion of patients that 
        recieved treatment at EOL among the different subgroup populations
    """
    impact_func = partial(get_eol_treatment_receival)
    si = SubgroupImpact(df, impact_func, subgroups=subgroups)
    summary = si.get_summary()
    for subgroup, res in summary.items():
        res.columns.name = None
        res.columns = pd.MultiIndex.from_product([[subgroup], res.columns])
    summary = pd.concat(summary.values(), axis=1)
    return summary
    
def get_eol_treatment_receival(group):
    counts = {}
    name = 'Received Treatment Near EOL'
    received = group['received_treatment_near_EOL'].value_counts()
    counts[name] = received.get(True, 0)
    counts[f'Not {name}'] = received.get(False, 0)
    total = received.sum()
    rate = counts[name] / total
    lower, upper = proportion_confint(counts[name], total)
    counts[f'{name} (Rate)'] = f'{rate:.3f} ({lower:.3f}-{upper:.3f})'
    return pd.Series(counts)
    
###############################################################################
# Subgroup Impact
###############################################################################
class SubgroupImpact(SubgroupSummary):
    def __init__(
        self, 
        data, 
        impact_func,
        subgroups=None,
        top=3,
        cohort_name='Test',
        **kwargs
    ):
        """
        Args:
            top (int): the number of most common categories. We only analyze 
                populations subgroups belonging to those top categories
            **kwargs: keyword arguments fed into SubgroupSummary
        """
        super().__init__(data, **kwargs)
        self.impact_func = impact_func
        if subgroups is None:
            subgroups = [
                'all', 'age', 'sex', 'immigration', 'language', 'arrival', 
                'income', 'area_density', 'regimen', 'cancer_location', 
                'days_since_starting'
            ]
        self.subgroups = subgroups
        self.top = top
        self.cohort_name = cohort_name
        
    def get_summary(self, *args):
        summary = {}
        if 'all' in self.subgroups: 
            self.entire_cohort_summary(summary, *args)
        if 'age' in self.subgroups: 
            self.age_summary(summary, *args)
        if 'sex' in self.subgroups: 
            self.sex_summary(summary, *args)
        if 'immigrant' in self.subgroups: 
            self.immigration_summary(summary, *args)
        if 'language' in self.subgroups: 
            self.language_summary(summary, *args)
        if 'arrival' in self.subgroups: 
            self.arrival_summary(summary, *args)
        if 'world_region_of_birth' in self.subgroups: 
            self.world_region_of_birth_summary(summary, *args)
        if 'income' in self.subgroups: 
            self.income_summary(summary, *args)
        if 'area_density' in self.subgroups: 
            self.area_density_summary(summary, *args)
        if 'regimen' in self.subgroups: 
            self.most_common_category_summary(
                summary, *args, catcol='regimen', 
                transform=str.upper, top=self.top
            )
        if 'cancer_location' in self.subgroups: 
            self.most_common_category_summary(
                summary, *args, catcol='cancer_topog_cd', 
                mapping=cancer_code_mapping, top=self.top
            )
        if 'days_since_starting' in self.subgroups: 
            self.days_since_starting_regimen_summary(summary, *args)
            
        for subgroup, result in summary.items():
            summary[subgroup] = pd.concat(result, names=[subgroup], axis=1)
            
        return summary

    def entire_cohort_summary(self, *args):
        mask = self.data['ikn'].notnull()
        self._summary(
            *args, mask=mask, subgroup=f'Entire {self.cohort_name} Cohort',
        )

    def _summary(
        self, 
        summary,
        mask=None, 
        subgroup='', 
        category=''
    ):
        if mask is None: raise ValueError('Please provide a mask')
        summary[subgroup] = summary.get(subgroup, {})
        summary[subgroup][category] = self.impact_func(self.data[mask])
