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
#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('cd', '../')
# reloads all modules everytime before cell is executed (no need to restart kernel)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 150)
import seaborn as sns

from src.config import root_path, can_folder, split_date, SCr_rise_threshold
from src.utility import (
    initialize_folders, load_predictions, 
    get_nunique_categories, get_nmissing, most_common_categories,
    get_hyperparameters
)
from src.summarize import data_description_summary, feature_summary
from src.visualize import importance_plot, subgroup_performance_plot
from src.prep_data import PrepDataCAN
from src.train import TrainML, TrainRNN, TrainENS
from src.evaluate import Evaluate


# In[3]:


processes = 64
target_keyword = 'SCr|dialysis|next'
main_dir = f'{root_path}/{can_folder}'


# # Acute Kidney Injury

# In[133]:


adverse_event = 'aki'
output_path = f'{main_dir}/models/{adverse_event.upper()}'
initialize_folders(output_path)


# ## Prepare Data for Model Training

# In[5]:


prep = PrepDataCAN(adverse_event=adverse_event, target_keyword=target_keyword)
model_data = prep.get_data(verbose=True)
model_data


# In[6]:


most_common_categories(model_data, catcol='regimen', with_respect_to='patients', top=10)


# In[7]:


sorted(model_data.columns.tolist())


# In[8]:


get_nunique_categories(model_data)


# In[9]:


get_nmissing(model_data)


# In[134]:


prep = PrepDataCAN(adverse_event=adverse_event, target_keyword=target_keyword) # need to reset
model_data = prep.get_data(missing_thresh=80, include_comorbidity=True, verbose=True)
X, Y, tag = prep.split_and_transform_data(model_data, split_date=split_date)
# remove sessions in model_data that were excluded during split_and_transform
model_data = model_data.loc[tag.index]


# In[107]:


prep.get_label_distribution(Y, tag, with_respect_to='sessions')


# In[12]:


prep.get_label_distribution(Y, tag, with_respect_to='patients')


# In[137]:


# Convenience variables
train_mask, valid_mask, test_mask = tag['split'] == 'Train', tag['split'] == 'Valid', tag['split'] == 'Test'
X_train, X_valid, X_test = X[train_mask], X[valid_mask], X[test_mask]
Y_train, Y_valid, Y_test = Y[train_mask], Y[valid_mask], Y[test_mask]


# ### Study Characteristics

# In[14]:


subgroups = [
    'sex', 'immigration', 'birth_region', 'language', 'income', 'area_density',
    'regimen', 'cancer_type', 'cancer_location', 'target', 'comorbidity', 'dialysis', 'ckd'
]
data_description_summary(
    model_data, Y, tag, save_dir=f'{output_path}/tables', partition_method='cohort', target_event='AKI', subgroups=subgroups
)


# ### Feature Characteristic

# In[15]:


df = prep.ohe.encode(model_data.copy(), verbose=False) # get original (non-normalized, non-imputed) data one-hot encoded
df = df[train_mask].drop(columns=['ikn'])
feature_summary(
    df, save_dir=f'{output_path}/tables', deny_old_survey=True, event_dates=prep.event_dates[train_mask]
).head(60)


# ## Train Models

# ### Main ML Models

# In[16]:


train_ml = TrainML(X, Y, tag, output_path, n_jobs=processes)
train_ml.tune_and_train(run_bayesopt=False, run_training=True, save_preds=True)


# ### RNN Model

# In[17]:


# Distrubution of the sequence lengths in the training set
dist_seq_lengths = X_train.groupby(tag.loc[train_mask, 'ikn']).apply(len)
dist_seq_lengths = dist_seq_lengths.clip(upper=dist_seq_lengths.quantile(q=0.999))
fig, ax = plt.subplots(figsize=(15, 3))
ax.grid(zorder=0)
sns.histplot(dist_seq_lengths, ax=ax, zorder=2, bins=int(dist_seq_lengths.max()))


# In[18]:


train_rnn = TrainRNN(X, Y, tag, output_path)
train_rnn.tune_and_train(run_bayesopt=False, run_training=True, run_calibration=True, save_preds=True)


# ### ENS Model 

# In[108]:


# combine rnn and ml predictions
preds = load_predictions(save_dir=f'{output_path}/predictions')
preds_rnn = load_predictions(save_dir=f'{output_path}/predictions', filename='rnn_predictions')
for split, pred in preds_rnn.items(): preds[split]['RNN'] = pred
del preds_rnn
# Initialize Training Class
train_ens = TrainENS(X, Y, tag, output_path, preds)


# In[109]:


train_ens.tune_and_train(run_bayesopt=False, run_calibration=False, calibrate_pred=True)


# ## Evaluate Models

# In[110]:


preds, labels = train_ens.preds, train_ens.labels


# In[22]:


eval_models = Evaluate(output_path=output_path, preds=preds, labels=labels, orig_data=model_data)


# In[23]:


baseline_cols = ['regimen', 'baseline_eGFR']
kwargs = {'baseline_cols': baseline_cols, 'display_ci': True, 'load_ci': False, 'save_ci': True}
eval_models.get_evaluation_scores(**kwargs)


# In[24]:


eval_models.plot_curves(curve_type='pr', legend_loc='lower left', save=False)
eval_models.plot_curves(curve_type='roc', legend_loc='lower right', save=False)
eval_models.plot_curves(curve_type='pred_cdf', save=False) # cumulative distribution function of model prediction
eval_models.plot_calibs(legend_loc='upper left', save=False) 
# eval_models.plot_calibs(include_pred_hist=True, legend_loc='upper left', figsize=(12,28), padding={'pad_y1': 0.3, 'pad_y0': 3.0})


# ## Post-Training Analysis

# ### Threshold Op Points

# In[25]:


pred_thresholds = np.arange(0.05, 0.51, 0.05)
perf_metrics = [
    'warning_rate', 'precision', 'recall', 'NPV', 'specificity', 
]
thresh_df = eval_models.operating_points(points=pred_thresholds, alg='ENS', op_metric='threshold', perf_metrics=perf_metrics)
thresh_df


# ### All the Plots

# In[26]:


eval_models.all_plots_for_single_target(alg='ENS', target_event='AKI')


# ### Most Important Features/Feature Groups

# In[27]:


get_ipython().system('python scripts/feat_imp.py --adverse-event AKI --output-path {output_path} ')


# In[28]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', eval_models.target_events, output_path, figsize=(6,5), top=10, importance_by='feature', padding={'pad_x0': 2.7})


# In[29]:


get_ipython().system('python scripts/feat_imp.py --adverse-event AKI --output-path {output_path} --permute-group')


# In[30]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', eval_models.target_events, output_path, figsize=(6,5), top=10, importance_by='group', padding={'pad_x0': 1.2})


# ### Performance on Subgroups

# In[31]:


subgroups = [
    'all', 'age', 'sex', 'immigrant', 'language', 'arrival', 'income', 
    'area_density', 'ckd', 'regimen', 'cancer_location', 'days_since_starting', 
]
perf_kwargs = {'perf_metrics': ['precision', 'recall', 'event_rate']}
subgroup_performance = eval_models.get_perf_by_subgroup(
    subgroups=subgroups, pred_thresh=0.1, alg='ENS', display_ci=True, load_ci=False, perf_kwargs=perf_kwargs
)
subgroup_performance


# In[32]:


subgroup_performance = pd.read_csv(f'{output_path}/tables/subgroup_performance.csv', index_col=[0,1], header=[0,1])
groupings = {
    'Demographic': ['Entire Test Cohort', 'Age', 'Sex', 'Immigration', 'Language', 'Neighborhood Income Quintile'],
    'Treatment': ['Entire Test Cohort', 'Regimen', 'Topography ICD-0-3', 'Days Since Starting Regimen', 'CKD Prior to Treatment']
}
padding = {'pad_y0': 1.2, 'pad_x1': 2.6, 'pad_y1': 0.2}
for name, subgroups in groupings.items():
    subgroup_performance_plot(
        subgroup_performance, target_event='AKI', subgroups=subgroups, padding=padding,
        figsize=(12,30), save_dir=f'{output_path}/figures/subgroup_performance/{name}'
    )
# PPV = 0.3 means roughly for every 3 alarms, 2 are false alarms and 1 is true alarm
# Sesnsitivity = 0.5 means roughly for every 2 true alarms, the model predicts 1 of them correctly
# Event Rate = 0.15 means true alarms occur 15% of the time


# ### Decision Curve Plot

# In[33]:


result = eval_models.plot_decision_curves('ENS')
result['AKI'].tail(n=100)


# In[34]:


get_hyperparameters(output_path)


# # Chronic Kidney Disease

# In[143]:


adverse_event = 'ckd'
output_path = f'{main_dir}/models/{adverse_event.upper()}'
initialize_folders(output_path)


# ## Prepare Data for Model Training

# In[144]:


prep = PrepDataCAN(adverse_event=adverse_event, target_keyword=target_keyword)
model_data = prep.get_data(missing_thresh=80, include_comorbidity=True, verbose=True)
model_data['next_eGFR'].hist(bins=100)
X, Y, tag = prep.split_and_transform_data(model_data, split_date=split_date)
# remove sessions in model_data that were excluded during split_and_transform
model_data = model_data.loc[tag.index]


# In[37]:


prep.get_label_distribution(Y, tag, with_respect_to='sessions')


# In[38]:


prep.get_label_distribution(Y, tag, with_respect_to='patients')


# In[145]:


# Convenience variables
train_mask, valid_mask, test_mask = tag['split'] == 'Train', tag['split'] == 'Valid', tag['split'] == 'Test'
X_train, X_valid, X_test = X[train_mask], X[valid_mask], X[test_mask]
Y_train, Y_valid, Y_test = Y[train_mask], Y[valid_mask], Y[test_mask]


# ### Study Characteristics

# In[40]:


subgroups = [
    'sex', 'immigration', 'birth_region', 'language', 'income', 'area_density',
    'regimen', 'cancer_type', 'cancer_location', 'target', 'comorbidity', 'dialysis', 'ckd'
]
data_description_summary(
    model_data, Y, tag, save_dir=f'{output_path}/tables', partition_method='cohort', target_event='CKD', subgroups=subgroups
)


# ### Feature Characteristic

# In[41]:


df = prep.ohe.encode(model_data.copy(), verbose=False) # get original (non-normalized, non-imputed) data one-hot encoded
df = df[train_mask].drop(columns=['ikn'])
feature_summary(
    df, save_dir=f'{output_path}/tables', deny_old_survey=True, event_dates=prep.event_dates[train_mask]
).head(60)


# ## Train Models

# ### Main ML Models

# In[42]:


train_ml = TrainML(X, Y, tag, output_path, n_jobs=processes)
train_ml.tune_and_train(run_bayesopt=False, run_training=True, save_preds=True)


# ### RNN Model

# In[43]:


# Distrubution of the sequence lengths in the training set
dist_seq_lengths = X_train.groupby(tag.loc[train_mask, 'ikn']).apply(len)
dist_seq_lengths = dist_seq_lengths.clip(upper=dist_seq_lengths.quantile(q=0.999))
fig, ax = plt.subplots(figsize=(15, 3))
ax.grid(zorder=0)
sns.histplot(dist_seq_lengths, ax=ax, zorder=2, bins=int(dist_seq_lengths.max()))


# In[44]:


train_rnn = TrainRNN(X, Y, tag, output_path)
train_rnn.tune_and_train(run_bayesopt=False, run_training=True, run_calibration=True, save_preds=True)


# ### ENS Model 

# In[148]:


# combine rnn and ml predictions
preds = load_predictions(save_dir=f'{output_path}/predictions')
preds_rnn = load_predictions(save_dir=f'{output_path}/predictions', filename='rnn_predictions')
for split, pred in preds_rnn.items(): preds[split]['RNN'] = pred
del preds_rnn
# Initialize Training Class
train_ens = TrainENS(X, Y, tag, output_path, preds)


# In[149]:


train_ens.tune_and_train(run_bayesopt=False, run_calibration=False, calibrate_pred=True)


# ## Evaluate Models

# In[150]:


preds, labels = train_ens.preds, train_ens.labels


# In[48]:


eval_models = Evaluate(output_path=output_path, preds=preds, labels=labels, orig_data=model_data)


# In[49]:


baseline_cols = ['regimen', 'baseline_eGFR']
kwargs = {'baseline_cols': baseline_cols, 'display_ci': True, 'load_ci': False, 'save_ci': True}
eval_models.get_evaluation_scores(**kwargs)


# In[50]:


eval_models.plot_curves(curve_type='pr', legend_loc='lower left', save=False)
eval_models.plot_curves(curve_type='roc', legend_loc='lower right', save=False)
eval_models.plot_curves(curve_type='pred_cdf', save=False) # cumulative distribution function of model prediction
eval_models.plot_calibs(legend_loc='upper left', save=False) 
# eval_models.plot_calibs(include_pred_hist=True, legend_loc='upper left', figsize=(12,28), padding={'pad_y1': 0.3, 'pad_y0': 3.0})


# ## Post-Training Analysis

# ### Threshold Op Points

# In[51]:


pred_thresholds = np.arange(0.05, 0.51, 0.05)
perf_metrics = [
    'warning_rate', 'precision', 'recall', 'NPV', 'specificity', 
]
thresh_df = eval_models.operating_points(points=pred_thresholds, alg='ENS', op_metric='threshold', perf_metrics=perf_metrics)
thresh_df


# ### All the Plots

# In[52]:


eval_models.all_plots_for_single_target(alg='ENS', target_event='CKD')


# ### Most Important Features/Feature Groups

# In[53]:


get_ipython().system('python scripts/feat_imp.py --adverse-event CKD --output-path {output_path} ')


# In[54]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', eval_models.target_events, output_path, figsize=(6,15), top=10, importance_by='feature', padding={'pad_x0': 2.7})


# In[55]:


get_ipython().system('python scripts/feat_imp.py --adverse-event CKD --output-path {output_path} --permute-group')


# In[56]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', eval_models.target_events, output_path, figsize=(6,15), top=10, importance_by='group', padding={'pad_x0': 1.2})


# ### Performance on Subgroups

# In[57]:


subgroups = [
    'all', 'age', 'sex', 'immigrant', 'language', 'arrival', 'income', 
    'area_density', 'ckd', 'regimen', 'cancer_location', 'days_since_starting', 
]
perf_kwargs = {'perf_metrics': ['precision', 'recall', 'event_rate']}
subgroup_performance = eval_models.get_perf_by_subgroup(
    subgroups=subgroups, pred_thresh=0.1, alg='ENS', display_ci=True, load_ci=False, perf_kwargs=perf_kwargs
)
subgroup_performance


# In[58]:


subgroup_performance = pd.read_csv(f'{output_path}/tables/subgroup_performance.csv', index_col=[0,1], header=[0,1])
groupings = {
    'Demographic': ['Entire Test Cohort', 'Age', 'Sex', 'Immigration', 'Language', 'Neighborhood Income Quintile'],
    'Treatment': ['Entire Test Cohort', 'Regimen', 'Topography ICD-0-3', 'Days Since Starting Regimen', 'CKD Prior to Treatment']
}
padding = {'pad_y0': 1.2, 'pad_x1': 2.6, 'pad_y1': 0.2}
for name, subgroups in groupings.items():
    subgroup_performance_plot(
        subgroup_performance, target_event='CKD', subgroups=subgroups, padding=padding,
        figsize=(12,30), save_dir=f'{output_path}/figures/subgroup_performance/{name}'
    )
# PPV = 0.3 means roughly for every 3 alarms, 2 are false alarms and 1 is true alarm
# Sesnsitivity = 0.5 means roughly for every 2 true alarms, the model predicts 1 of them correctly
# Event Rate = 0.15 means true alarms occur 15% of the time


# ### Decision Curve Plot

# In[59]:


result = eval_models.plot_decision_curves('ENS')
result['CKD'].tail(n=100)


# In[60]:


get_hyperparameters(output_path)


# # Scratch Notes

# ## CKD + AKI Summaries

# In[61]:


from src.prep_data import PrepData
aki_prep = PrepDataCAN(adverse_event='aki', target_keyword=target_keyword)
ckd_prep = PrepDataCAN(adverse_event='ckd', target_keyword=target_keyword)

# get the union of ckd and aki dataset
ckd_data = ckd_prep.get_data(missing_thresh=80, include_comorbidity=True)
aki_data = aki_prep.get_data(missing_thresh=80, include_comorbidity=True)
df = pd.concat([aki_data, ckd_data])
df = df.reset_index().drop_duplicates(subset=['index']).set_index('index')

# set up a new prep object and combine the event_dates
prep = PrepData(target_keyword=target_keyword)
event_dates = pd.concat([aki_prep.event_dates, ckd_prep.event_dates])
event_dates = event_dates.reset_index().drop_duplicates(subset=['index']).set_index('index')
# aki and ckd may have different patient first visit dates in their dataset
# take the earlier date as the first visit date
event_dates['ikn'] = df['ikn']
patient_first_visit_date = event_dates.groupby('ikn')['visit_date'].min()
event_dates['first_visit_date'] = event_dates['ikn'].map(patient_first_visit_date)
prep.event_dates = event_dates

# ckd may not have same missingness variable as aki, which will cause ckd samples to have NaN in those missingness entries
# fill up missing missingness entries
cols = df.columns
cols = cols[cols.str.contains('_is_missing')]
df[cols] = df[cols].fillna(False) 

# set up the data
create_labels = lambda target: pd.concat([aki_prep.convert_labels(target), ckd_prep.convert_labels(target)], axis=1)
kwargs = {'split_date': split_date, 'impute': False, 'normalize': False, 'verbose': False, 'ohe_kwargs': {'verbose': False}}
X, Y, tag = prep.split_and_transform_data(df, **kwargs)
Y = create_labels(Y)
df = df.loc[tag.index]
train_mask = tag['split'] == 'Train'


# In[62]:


subgroups = [
    'sex', 'immigration', 'birth_region', 'language', 'income', 'area_density',
    'regimen', 'cancer_type', 'cancer_location', 'target', 'comorbidity', 'dialysis', 'ckd'
]
data_description_summary(
    df, Y, tag, save_dir=f'{main_dir}/models', partition_method='cohort', target_event='CKD', subgroups=subgroups
)


# In[63]:


df = prep.ohe.encode(df.copy(), verbose=False) # get original (non-normalized, non-imputed) data one-hot encoded
df = df[train_mask].drop(columns=['ikn'])
feature_summary(
    df, save_dir=f'{main_dir}/models', deny_old_survey=True, event_dates=prep.event_dates[train_mask]
).head(60)


# ## Spline Baseline Model

# In[64]:


from sklearn.preprocessing import StandardScaler
from src.train import TrainLOESSModel, TrainPolynomialModel
from src.evaluate import EvaluateBaselineModel
from src.visualize import get_bbox


# In[70]:


class BaselinePipeline():
    def __init__(self, event='ckd', alg='SPLINE'):
        Trains = {'LOESS': TrainLOESSModel, 'SPLINE': TrainPolynomialModel, 'POLY': TrainPolynomialModel}
        self.event = event
        self.alg = alg
        self.Train = Trains[alg]
        self.base_col = 'baseline_eGFR'
        self.regimen_subgroups = {'ALL', 'cisp(rt)', 'cisp(rt-w)'}
        self.output_path = f'{root_path}/{can_folder}/models/{event.upper()}'
        self.target_keyword = 'SCr|dialysis|next'
        
    def get_dataset(self):
        prep = PrepDataCAN(adverse_event=self.event, target_keyword=self.target_keyword)
        model_data = prep.get_data(missing_thresh=80, include_comorbidity=True)
        X, Y, tag = prep.split_and_transform_data(
            model_data, split_date=split_date, clip=False, verbose=False, ohe_kwargs={'verbose': False}
        )
        model_data = model_data.loc[tag.index]
        return model_data, X, Y, tag
    
    def train_and_eval_model(
        self, 
        X, 
        Y,
        tag, 
        split='Test', 
        name=None, 
        task_type='classification', 
        best_param_filename=''
    ):
        if name is None: name = self.event.upper()
        print(f'Training {self.alg} for {name}')
        
        train = self.Train(X, Y, tag, self.output_path, base_col=self.base_col, alg=self.alg, task_type=task_type)
        best_param = train.bayesopt(filename=best_param_filename, verbose=0)
        model = train.train_model(**best_param)

        print(f'Evaluating {self.alg} for {name}')
        preds, preds_min, preds_max = train.predict(model, split=split)

        return preds, preds_min, preds_max
    
    def run(self, split='Test'):
        orig_data, X, Y, tag = self.get_dataset()
        preds, preds_min, preds_max = self.train_and_eval_model(X, Y, tag, split=split)
        mask = tag['split'] == split
        data = orig_data[mask]
        Y = Y[mask]
        
        for i, regimen in enumerate(self.regimen_subgroups):
            df = data if regimen == 'ALL' else data[data['regimen'] == regimen]
            idxs = df.index

            predictions, labels = {split: {self.alg: preds.loc[idxs]}},  {split: Y.loc[idxs]}
            eval_loess = EvaluateBaselineModel(
                base_col=self.base_col, preds_min=preds_min.loc[idxs], preds_max=preds_max.loc[idxs], 
                output_path=self.output_path, preds=predictions, labels=labels, orig_data=df
            )
        
            print(f'{self.alg} plot for regimen {regimen}')
            eval_loess.all_plots(alg=self.alg, filename=f'{regimen}_{self.alg}')
            plt.show()
    
class RegressionBaselinePipeline(BaselinePipeline):
    def __init__(self, event='next_eGFR', alg='SPLINE'):
        super().__init__(event='ckd', alg=alg)
        self.reg_event = event
        self.name = 'Next eGFR'
    
    def scale_targets(self, Y, tag):
        scaler = StandardScaler()
        Y[tag['split'] == 'Train'] = scaler.fit_transform(Y[tag['split'] == 'Train'])
        Y[tag['split'] == 'Valid'] = scaler.transform(Y[tag['split'] == 'Valid'])
        Y[tag['split'] == 'Test'] = scaler.transform(Y[tag['split'] == 'Test'])
        return Y, scaler
    
    def inverse_scale_preds(self, predictions, scaler):
        preds, preds_min, preds_max = predictions
        preds[:] = scaler.inverse_transform(preds)
        preds_min[:] = scaler.inverse_transform(preds_min)
        preds_max[:] = scaler.inverse_transform(preds_max)
        return preds, preds_min, preds_max

    def run(self, split='Test'):
        orig_data, X, Y, tag = self.get_dataset()
        Y, scaler = self.scale_targets(orig_data[[self.reg_event]].copy(), tag)
        kwargs = {'split': split, 'name': self.name, 'task_type': 'regression', 
                  'best_param_filename': f'{self.alg}_regressor_best_param'}
        predictions = self.train_and_eval_model(X, Y, tag, **kwargs)
        (preds, preds_min, preds_max) = self.inverse_scale_preds(predictions, scaler)
        
        mask = tag['split'] == split
        data = orig_data[mask]
        Y = Y[mask]
        
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,12))
        axes = axes.flatten()
        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        
        for i, regimen in enumerate(self.regimen_subgroups):
            df = data if regimen == 'ALL' else data[data['regimen'] == regimen]
            idxs = df.index

            predictions, labels = {split: {self.alg: preds.loc[idxs]}},  {split: Y.loc[idxs]}
            eval_loess = EvaluateBaselineModel(
                base_col=self.base_col, preds_min=preds_min.loc[idxs], preds_max=preds_max.loc[idxs], 
                output_path=self.output_path, preds=predictions, labels=labels, orig_data=df
            )
        
            eval_loess.plot_loess(axes[i], self.alg, self.reg_event, split=split)

            filename = f'{self.output_path}/figures/baseline/{self.reg_event}_{regimen}_{self.alg}.jpg'
            fig.savefig(filename, bbox_inches=get_bbox(axes[i], fig), dpi=300) 
            axes[i].set_title(regimen)
            
        filename = f'{self.output_path}/figures/baseline/{self.reg_event}_{self.alg}.jpg'
        plt.savefig(filename, bbox_inches='tight', dpi=300)


# In[71]:


pipeline = BaselinePipeline(event='ckd')
pipeline.run()


# In[72]:


pipeline = RegressionBaselinePipeline(event='next_eGFR')
pipeline.run()


# ### Save the CKD Spline Baseline Model as a Threshold Table

# In[74]:


pipeline = BaselinePipeline(event='ckd')
orig_data, X, Y, tag = pipeline.get_dataset()
preds, preds_min, preds_max = pipeline.train_and_eval_model(X, Y, tag, split='Train')


# In[80]:


base_col = pipeline.base_col
base = orig_data.loc[preds.index, base_col]
df = pd.concat([preds.round(3), base.round(1)], axis=1).sort_values(by=base_col)
df = df.drop_duplicates(base_col)
df.to_csv(f'{output_path}/SPLINE_model.csv', index=False)


# ## Motwani Score Based Model

# In[96]:


prep = PrepDataCAN(adverse_event='aki', target_keyword=target_keyword)
df = prep.get_data(include_comorbidity=True)
X, Y, tag = prep.split_and_transform_data(df, split_date=split_date, verbose=False, ohe_kwargs={'verbose': False})
print(f'Size of data = {len(df)}, Number of patients = {df["ikn"].nunique()}')
df['cisplatin_dosage'] *= df['body_surface_area'] # convert from mg/m^2 to mg
df = df.loc[tag['split']=='Test']
print(f'Size of test data = {len(df)}, Number of patients = {df["ikn"].nunique()}')
df = df[df['baseline_albumin_value'].notnull()]
print(f'Size of test data with albumin = {len(df)}, Number of patients = {df["ikn"].nunique()}')
df = df.query('days_since_starting_chemo == 0') # very first treatment
print(f'Size of test data with only first day chemos = {len(df)}, Number of patients = {df["ikn"].nunique()}')


# In[97]:


def compute_score(data):
    score = pd.Series(0, index=data.index)
    score[data['age'].between(61, 70)] += 1.5
    score[data['age'] > 70] += 2.5
    score[data['baseline_albumin_value'] < 35] += 2.0
    score[data['cisplatin_dosage'].between(101, 150)] += 1.0
    score[data['cisplatin_dosage'] > 150] += 3.0
    score[data['hypertension']] += 2.0
    score /= score.max()
    return score


# In[113]:


score = compute_score(df)
labels = {'Test': Y.loc[df.index]}
preds = {'Test': {'ENS': train_ens.preds['Test']['ENS'].loc[df.index], 'MSB': pd.DataFrame({'AKI': score})}}
eval_motwani_model = Evaluate(output_path='', preds=preds, labels=labels, orig_data=df)


# In[115]:


# label distribtuion
labels['Test'].apply(pd.value_counts)


# In[118]:


kwargs = {'algs': ['ENS', 'MSB'], 'splits': ['Test'], 'display_ci': True, 'save_score': False}
result = eval_motwani_model.get_evaluation_scores(**kwargs)
result


# In[120]:


eval_motwani_model.all_plots_for_single_target(alg='MSB', target_event='AKI', n_bins=20, figsize=(12,16), save=False)


# In[126]:


points = np.arange(0, 8.6, 0.5)/8.5 # 8.5 is the highest score possible, 0 is lowest score possible
eval_motwani_model.operating_points(
    points, op_metric='threshold', alg='MSB', target_events=['AKI'], 
    perf_metrics=['warning_rate', 'precision', 'recall', 'NPV', 'specificity'], save=False
)


# ### Compare with ENS

# In[127]:


eval_motwani_model.all_plots_for_single_target(alg='ENS', target_event='AKI', n_bins=20,figsize=(12,16), save=False)


# In[128]:


points = np.arange(0, 8.6, 0.5)/8.5 # 8.5 is the highest score possible, 0 is lowest score possible
eval_motwani_model.operating_points(
    points, op_metric='threshold', alg='ENS', target_events=['AKI'], 
    perf_metrics=['warning_rate', 'precision', 'recall', 'NPV', 'specificity'], save=False
)


# ## Missingness By Splits

# In[135]:


from src.utility import get_nmissing_by_splits


# In[141]:


# Acute Kidney Injury
missing = get_nmissing_by_splits(model_data, train_ens.labels)
missing.sort_values(by=(f'Test (N={sum(test_mask)})', 'Missing (N)'), ascending=False)


# In[151]:


# Chronic Kidney Disease
missing = get_nmissing_by_splits(model_data, train_ens.labels)
missing.sort_values(by=(f'Test (N={sum(test_mask)})', 'Missing (N)'), ascending=False)
