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

# In[2]:


get_ipython().run_line_magic('cd', '../')
# reloads all modules everytime before cell is executed (no need to restart kernel)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


import os
import tqdm
import pickle
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 150)
import numpy as np
import matplotlib.pyplot as plt

from src.config import (root_path, can_folder, split_date, SCr_rise_threshold)
from src.utility import (initialize_folders, load_predictions, 
                         get_nunique_entries, get_nmissing, most_common_by_category,
                         get_hyperparameters)
from src.summarize import (data_characteristic_summary, feature_summary, subgroup_performance_summary)
from src.visualize import (importance_plot, subgroup_performance_plot)
from src.prep_data import (PrepData, PrepDataCAN)
from src.train import (TrainML, TrainRNN, TrainENS)
from src.evaluate import (Evaluate)


# In[4]:


processes = 64
target_keyword = 'SCr|dialysis|next'
main_dir = f'{root_path}/{can_folder}'


# # Acute Kidney Injury

# In[4]:


adverse_event = 'aki'
output_path = f'{main_dir}/models/{adverse_event.upper()}'
initialize_folders(output_path)


# ## Prepare Data for Model Training

# In[5]:


# Preparing Data for Model Input
prep = PrepDataCAN(adverse_event=adverse_event)


# In[6]:


model_data = prep.get_data(verbose=True)
model_data


# In[7]:


most_common_by_category(model_data, category='regimen', with_respect_to='patients', top=10)


# In[8]:


sorted(model_data.columns.tolist())


# In[9]:


get_nunique_entries(model_data)


# In[10]:


get_nmissing(model_data)


# In[11]:


model_data = prep.get_data(missing_thresh=80, verbose=True)
print(f"Size of model_data: {model_data.shape}")
print(f"Number of unique patients: {model_data['ikn'].nunique()}")
mask = (model_data['SCr_rise'] >= SCr_rise_threshold) | (model_data['SCr_fold_increase'] > 1.5)
N = model_data.loc[mask, 'ikn'].nunique()
print(f"Number of unique patients that had Acute Kidney Injury (AKI) " +\
      f"within 28 days or right before treatment session: {N}")


# In[12]:


kwargs = {'target_keyword': target_keyword, 'split_date': split_date}
# NOTE: any changes to X_train, X_valid, etc will also be seen in dataset
dataset = X_train, X_valid, X_test, Y_train, Y_valid, Y_test = prep.split_data(prep.dummify_data(model_data.copy()), **kwargs)


# In[13]:


prep.get_label_distribution(Y_train, Y_valid, Y_test)


# ## Train ML Models

# In[14]:


# Initialize Training class
train_ml = TrainML(dataset, output_path, n_jobs=processes)


# In[15]:


skip_alg = []
train_ml.tune_and_train(run_bayesopt=False, run_training=True, save_preds=True, skip_alg=skip_alg)


# ## Train RNN Model

# In[14]:


# Include ikn to the input data (recall that any changes to X_train, X_valid, etc will also be seen in dataset)
X_train['ikn'] = model_data['ikn']
X_valid['ikn'] = model_data['ikn']
X_test['ikn'] = model_data['ikn']

# Initialize Training class 
train_rnn = TrainRNN(dataset, output_path)


# In[15]:


train_rnn.tune_and_train(run_bayesopt=True, run_training=True, run_calibration=True, 
                         calibrate_pred=True, save_preds=True)


# ## Train ENS Model 

# In[17]:


# combine rnn and ml predictions
preds = load_predictions(save_dir=f'{output_path}/predictions')
preds_rnn = load_predictions(save_dir=f'{output_path}/predictions', filename='rnn_predictions')
for split, pred in preds_rnn.items(): preds[split]['RNN'] = pred
del preds_rnn
# Initialize Training Class
train_ens = TrainENS(dataset, output_path, preds)


# In[18]:


train_ens.tune_and_train(run_bayesopt=False, run_calibration=False, calibrate_pred=True)


# ## Evaluate Models

# In[19]:


eval_models = Evaluate(output_path=output_path, preds=train_ens.preds, labels=train_ens.labels, orig_data=model_data)


# In[52]:


baseline_cols = ['regimen', 'baseline_eGFR']
kwargs = {'get_baseline': True, 'baseline_cols': baseline_cols, 'display_ci': True, 'load_ci': True, 'save_ci': False, 'verbose': False}
eval_models.get_evaluation_scores(**kwargs)


# In[53]:


eval_models.plot_curves(curve_type='pr', legend_location='lower left', figsize=(12,18), save=False)
eval_models.plot_curves(curve_type='roc', legend_location='lower right', figsize=(12,18), save=False)
eval_models.plot_curves(curve_type='pred_cdf', figsize=(12,18), save=False) # cumulative distribution function of model prediction
eval_models.plot_calibs(legend_location='upper left', figsize=(12,18), save=False) 
# eval_models.plot_calibs(include_pred_hist=True, legend_location='upper left', figsize=(12,28), padding={'pad_y1': 0.3, 'pad_y0': 3.0})


# ## Post-Training Analysis

# ### Study Population Characteristics

# In[90]:


data_characteristic_summary(eval_models, save_dir=f'{output_path}/tables', partition='cohort', target_event='AKI',
                            include_combordity=True, include_ckd=True, include_dialysis=True)


# ### Feature Characteristic

# In[70]:


feature_summary(eval_models, prep, target_keyword=target_keyword, save_dir=f'{output_path}/tables').head(60)


# ### Threshold Op Points

# In[56]:


pred_thresholds = np.arange(0, 1.01, 0.05)
thresh_df = eval_models.operating_points(algorithm='ENS', points=pred_thresholds, metric='threshold')
thresh_df


# ### All the Plots

# In[57]:


eval_models.all_plots_for_single_target(algorithm='ENS', target_event='AKI', save=True)


# ### Most important features

# In[ ]:


get_ipython().system('python scripts/perm_importance.py --adverse-event CAAKI')


# In[17]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', eval_models.target_events, output_path, figsize=(6,5), top=10, importance_by='feature', padding={'pad_x0': 2.7})


# ### Performance on Subgroups

# In[20]:


subgroups = {'all', 'age', 'sex', 'immigrant', 'income', 'regimen', 'cancer_location', 'days_since_starting', 'ckd'}
df = subgroup_performance_summary('ENS', eval_models, pred_thresh=0.1, subgroups=subgroups, 
                                  display_ci=False, load_ci=False, save_ci=False)
df # @ pred threshold = 0.1


# In[30]:


# PPV = 0.3 means roughly for every 3 alarms, 2 are false alarms and 1 is true alarm
# Sesnsitivity = 0.5 means roughly for every 2 true alarms, the model predicts 1 of them correctly
# Event Rate = 0.15 means true alarms occur 15% of the time
groupings = {'Demographic': ['Entire Test Cohort', 'Age', 'Sex', 'Immigration', 'Neighborhood Income Quintile'],
             'Treatment': ['Entire Test Cohort', 'Regimen', 'Cancer Location', 'Days Since Starting Regimen', 'CKD Prior to Treatment']}
padding = {'pad_y0': 1.2, 'pad_x1': 2.6, 'pad_y1': 0.2}
for name, subgroups in groupings.items():
    subgroup_performance_plot(df, target_event='AKI', subgroups=subgroups, padding=padding,
                              figsize=(12,24), save=True, save_dir=f'{output_path}/figures/subgroup_performance/{name}')


# ### Decision Curve Plot

# In[78]:


_ = eval_models.plot_decision_curve_analysis('ENS')


# In[79]:


get_hyperparameters(output_path)


# # Chronic Kidney Disease

# In[31]:


adverse_event = 'ckd'
output_path = f'{main_dir}/models/{adverse_event.upper()}'
initialize_folders(output_path)


# ## Prepare Data for Model Training

# In[32]:


prep = PrepDataCAN(adverse_event=adverse_event)
model_data = prep.get_data(missing_thresh=80, verbose=True)
model_data['next_eGFR'].hist(bins=100)
print(f"Size of model_data: {model_data.shape}")
print(f"Number of unique patients: {model_data['ikn'].nunique()}")
mask = (model_data['next_eGFR'] < 60) | model_data['dialysis']
N = model_data.loc[mask, 'ikn'].nunique()
print(f"Number of unique patients that had Chronic Kidney Disease (CKD): {N}")
kwargs = {'target_keyword': target_keyword, 'split_date': split_date}
# NOTE: any changes to X_train, X_valid, etc will also be seen in dataset
dataset = X_train, X_valid, X_test, Y_train, Y_valid, Y_test = prep.split_data(prep.dummify_data(model_data.copy()), **kwargs)


# In[33]:


prep.get_label_distribution(Y_train, Y_valid, Y_test)


# ## Train ML Models

# In[19]:


# Initialize Training class
train_ml = TrainML(dataset, output_path, n_jobs=processes)


# In[20]:


skip_alg = []
train_ml.tune_and_train(run_bayesopt=False, run_training=True, save_preds=True, skip_alg=skip_alg)


# ## Train RNN Model

# In[20]:


# Include ikn to the input data (recall that any changes to X_train, X_valid, etc will also be seen in dataset)
X_train['ikn'] = model_data['ikn']
X_valid['ikn'] = model_data['ikn']
X_test['ikn'] = model_data['ikn']

# Initialize Training class 
train_rnn = TrainRNN(dataset, output_path)


# In[21]:


train_rnn.tune_and_train(run_bayesopt=True, run_training=True, run_calibration=True, save_preds=True)


# ## Train ENS Model 

# In[34]:


# combine rnn and ml predictions
preds = load_predictions(save_dir=f'{output_path}/predictions')
preds_rnn = load_predictions(save_dir=f'{output_path}/predictions', filename='rnn_predictions')
for split, pred in preds_rnn.items(): preds[split]['RNN'] = pred
del preds_rnn
# Initialize Training Class
train_ens = TrainENS(dataset, output_path, preds)


# In[35]:


train_ens.tune_and_train(run_bayesopt=False, run_calibration=False, calibrate_pred=True)


# ## Evaluate Models

# In[36]:


eval_models = Evaluate(output_path=output_path, preds=train_ens.preds, labels=train_ens.labels, orig_data=model_data)


# In[99]:


baseline_cols = ['regimen', 'baseline_eGFR']
kwargs = {'get_baseline': True, 'baseline_cols': baseline_cols, 'display_ci': True, 'load_ci': False, 'save_ci': True, 'verbose': False}
eval_models.get_evaluation_scores(**kwargs)


# In[100]:


eval_models.plot_curves(curve_type='pr', legend_location='upper right', figsize=(12,18))
eval_models.plot_curves(curve_type='roc', legend_location='lower right', figsize=(12,18))
eval_models.plot_curves(curve_type='pred_cdf', figsize=(12,18)) # cumulative distribution function of model prediction
eval_models.plot_calibs(legend_location='upper left', figsize=(12,18)) 
# eval_models.plot_calibs(include_pred_hist=True, legend_location='upper left', figsize=(12,28), padding={'pad_y1': 0.3, 'pad_y0': 3.0})


# ## Post-Training Analysis

# ### Study Population Characteristics

# In[101]:


data_characteristic_summary(eval_models, save_dir=f'{output_path}/tables', partition='cohort', target_event='CKD',
                            include_combordity=True, include_ckd=True, include_dialysis=True)


# ### Feature Characteristic

# In[102]:


feature_summary(eval_models, prep, target_keyword=target_keyword, save_dir=f'{output_path}/tables').head(60)


# ### Threshold Op Points

# In[103]:


pred_thresholds = np.arange(0, 1.01, 0.05)
thresh_df = eval_models.operating_points(algorithm='ENS', points=pred_thresholds, metric='threshold')
thresh_df


# ### All the Plots

# In[104]:


eval_models.all_plots_for_single_target(algorithm='ENS', target_event='CKD', save=True)


# ### Most important features

# In[ ]:


get_ipython().system('python scripts/perm_importance.py --adverse-event CACKD')


# In[24]:


# importance score is defined as the decrease in AUROC Score when feature value is randomly shuffled
importance_plot('ENS', eval_models.target_events, output_path, figsize=(6,15), top=10, importance_by='feature', padding={'pad_x0': 2.7})


# ### Performance on Subgroups

# In[37]:


subgroups = {'all', 'age', 'sex', 'immigrant', 'income', 'regimen', 'cancer_location', 'days_since_starting', 'ckd'}
df = subgroup_performance_summary('ENS', eval_models, pred_thresh=[0.4, 0.25, 0.1], 
                                  subgroups=subgroups, display_ci=False, load_ci=False, save_ci=False)
df # @ pred threshold = 0.4


# In[38]:


# PPV = 0.3 means roughly for every 3 alarms, 2 are false alarms and 1 is true alarm
# Sesnsitivity = 0.5 means roughly for every 2 true alarms, the model predicts 1 of them correctly
# Event Rate = 0.15 means true alarms occur 15% of the time
groupings = {'Demographic': ['Entire Test Cohort', 'Age', 'Sex', 'Immigration', 'Neighborhood Income Quintile'],
             'Treatment': ['Entire Test Cohort', 'Regimen', 'Cancer Location', 'Days Since Starting Regimen', 'CKD Prior to Treatment']}
padding = {'pad_y0': 1.2, 'pad_x1': 2.6, 'pad_y1': 0.2}
for name, subgroups in groupings.items():
    for target_event in eval_models.target_events:
        print(f'Plotted {name} Subgroup Performance Plot for {target_event}')
        subgroup_performance_plot(df, target_event=target_event, subgroups=subgroups, padding=padding,
                                  figsize=(12,24), save=True, save_dir=f'{output_path}/figures/subgroup_performance/{name}')


# ### Decision Curve Plot

# In[107]:


eval_models.plot_decision_curve_analysis('ENS')


# In[108]:


get_hyperparameters(output_path)


# # Scratch Notes

# ## CKD + AKI Summaries

# In[127]:


from src.prep_data import PrepData
aki_prep = PrepDataCAN(adverse_event='aki')
ckd_prep = PrepDataCAN(adverse_event='ckd')

# get the union of ckd and aki dataset
ckd_data = ckd_prep.get_data(missing_thresh=80)
aki_data = aki_prep.get_data(missing_thresh=80)
df = pd.concat([aki_data, ckd_data])
df = df.reset_index().drop_duplicates(subset=['index']).set_index('index')

# set up a new prep object and combine the event_dates
prep = PrepData()
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

# set up the labels
create_labels = lambda target: pd.concat([aki_prep.convert_labels(target), ckd_prep.convert_labels(target)], axis=1)
kwargs = {'target_keyword': target_keyword, 'split_date': split_date, 'impute': False, 
          'normalize': False, 'verbose': False}
_, _, _, Y_train, Y_valid, Y_test = prep.split_data(df, **kwargs)
labels = {'Train': create_labels(Y_train), 'Valid': create_labels(Y_valid), 'Test': create_labels(Y_test)}

# set up the Evaluate object
pred_placeholder = {'Test': {}}
eval_models = Evaluate(output_path='placeholder', preds=pred_placeholder, labels=labels, orig_data=df)


# In[128]:


data_characteristic_summary(eval_models, save_dir=f'{main_dir}/models', partition='cohort', 
                            include_combordity=True, include_ckd=True, include_dialysis=True)


# In[129]:


feature_summary(eval_models, prep, target_keyword=target_keyword, save_dir=f'{main_dir}/models').head(60)


# ## Spline Baseline Model

# In[13]:


from sklearn.preprocessing import StandardScaler
from src.train import TrainLOESSModel, TrainPolynomialModel
from src.evaluate import EvaluateBaselineModel
from src.visualize import get_bbox


# In[28]:


class End2EndPipeline():
    def __init__(self, event='ckd', algorithm='SPLINE'):
        Trains = {'LOESS': TrainLOESSModel, 'SPLINE': TrainPolynomialModel, 'POLY': TrainPolynomialModel}
        self.event = event
        self.algorithm = algorithm
        self.Train = Trains[algorithm]
        self.base_col = base_col = 'baseline_eGFR'
        self.regimen_subgroups = {'ALL', 'cisp(rt)', 'cisp(rt-w)'}
        self.output_path = f'{root_path}/{can_folder}/models/{event.upper()}'
        self.target_keyword = 'SCr|dialysis|next'
        
    def get_dataset(self):
        prep = PrepDataCAN(adverse_event=self.event)
        data = prep.get_data(missing_thresh=80)
        data, clip_thresholds = prep.clip_outliers(data, lower_percentile=0.001, upper_percentile=0.999)
        kwargs = {'target_keyword': self.target_keyword, 'split_date': split_date, 'verbose': False}
        dataset = prep.split_data(prep.dummify_data(data.copy()), **kwargs)
        return data, dataset
    
    def train_and_eval_model(self, dataset, split='Test', name=None, 
                             task_type='classification', best_param_filename=''):
        if name is None: name = self.event.upper()
        print(f'Training {self.algorithm} for {name}')
        train = self.Train(dataset, self.output_path, base_col=self.base_col, algorithm=self.algorithm, task_type=task_type)
        best_param = train.bayesopt(filename=best_param_filename, verbose=0)
        model = train.train_model(**best_param)

        print(f'Evaluating {self.algorithm} for {name}')
        (preds, preds_min, preds_max), Y = train.predict(model, split=split)

        return (preds, preds_min, preds_max), Y
    
    def run(self, split='Test'):
        data, dataset = self.get_dataset()
        (preds, preds_min, preds_max), Y = self.train_and_eval_model(dataset, split=split)
        data = data.loc[Y.index]
        for i, regimen in enumerate(self.regimen_subgroups):
            df = data if regimen == 'ALL' else data[data['regimen'] == regimen]
            idxs = df.index

            predictions, labels = {split: {self.algorithm: preds.loc[idxs]}},  {split: Y.loc[idxs]}
            eval_loess = EvaluateBaselineModel(base_col=self.base_col, preds_min=preds_min.loc[idxs], preds_max=preds_max.loc[idxs], 
                                               output_path=self.output_path, preds=predictions, labels=labels, orig_data=df)
        
            print(f'{self.algorithm} plot for regimen {regimen}')
            eval_loess.all_plots(algorithm=self.algorithm, filename=f'{regimen}_{self.algorithm}')
            
        return data, preds
    
class End2EndPipelineRegression(End2EndPipeline):
    def __init__(self, event='next_eGFR', algorithm='SPLINE'):
        super().__init__(event='ckd', algorithm=algorithm)
        self.reg_event = event
        self.name = 'Next eGFR'
    
    def scale_targets(self, dataset, data):
        cols = [self.reg_event]
        X_train, X_valid, X_test, _, _, _ = dataset
        Y_train, Y_valid, Y_test =  data.loc[X_train.index, cols], data.loc[X_valid.index, cols], data.loc[X_test.index, cols]

        scaler = StandardScaler()
        Y_train[:] = scaler.fit_transform(Y_train)
        Y_valid[:] = scaler.transform(Y_valid)
        Y_test[:] = scaler.transform(Y_test)

        dataset = (X_train, X_valid, X_test, Y_train, Y_valid, Y_test)
        return dataset, scaler
    
    def inverse_scale_preds(self, predictions, scaler):
        preds, preds_min, preds_max = predictions
        preds[:] = scaler.inverse_transform(preds)
        preds_min[:] = scaler.inverse_transform(preds_min)
        preds_max[:] = scaler.inverse_transform(preds_max)
        return preds, preds_min, preds_max

    def run(self, split='Test'):
        data, dataset = self.get_dataset()
        dataset, scaler = self.scale_targets(dataset, data)
        kwargs = {'split': split, 'name': self.name, 'task_type': 'regression', 
                  'best_param_filename': f'{self.algorithm}_regressor_best_param'}
        predictions, Y = self.train_and_eval_model(dataset, **kwargs)
        (preds, preds_min, preds_max) = self.inverse_scale_preds(predictions, scaler)
        
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,12))
        axes = axes.flatten()
        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        
        data = data.loc[Y.index]
        for i, regimen in enumerate(self.regimen_subgroups):
            df = data if regimen == 'ALL' else data[data['regimen'] == regimen]
            idxs = df.index

            predictions, labels = {split: {self.algorithm: preds.loc[idxs]}},  {split: Y.loc[idxs]}
            eval_loess = EvaluateBaselineModel(base_col=self.base_col, preds_min=preds_min.loc[idxs], preds_max=preds_max.loc[idxs], 
                                               output_path=self.output_path, preds=predictions, labels=labels, orig_data=df)
        
            eval_loess.plot_loess(axes[i], self.algorithm, self.reg_event, split=split)

            filename = f'{self.output_path}/figures/baseline/{self.reg_event}_{regimen}_{self.algorithm}.jpg'
            fig.savefig(filename, bbox_inches=get_bbox(axes[i], fig), dpi=300) 
            axes[i].set_title(regimen)
            
        filename = f'{self.output_path}/figures/baseline/{self.reg_event}_{self.algorithm}.jpg'
        plt.savefig(filename, bbox_inches='tight', dpi=300)


# In[21]:


pipeline = End2EndPipeline(event='ckd')
pipeline.run()


# In[22]:


pipeline = End2EndPipeline(event='aki')
pipeline.run()


# In[29]:


pipeline = End2EndPipelineRegression(event='next_eGFR')
pipeline.run()


# ### Save the CKD Spline Baseline Model as a Threshold Table

# In[30]:


pipeline = End2EndPipeline(event='ckd')
data, dataset = pipeline.get_dataset()
(preds, preds_min, preds_max), Y = pipeline.train_and_eval_model(dataset, split='Train')


# In[62]:


df = pd.concat([preds, data.loc[preds.index, pipeline.base_col]], axis=1)
df = df.sort_values('baseline_eGFR')
df = df.round(3)
df['baseline_eGFR'] = df['baseline_eGFR'].round(1)
df = df.drop_duplicates('baseline_eGFR')
df.to_csv(f'{output_path}/SPLINE_model.csv', index=False)


# ## Motwani Score Based Model

# In[144]:


df = prep.get_data()
print(f'Size of data = {len(df)}, Number of patients = {df["ikn"].nunique()}')
df['cisplatin_dosage'] *= df['body_surface_area'] # convert from mg/m^2 to mg
df = df.loc[Y_test.index]
print(f'Size of test data = {len(df)}, Number of patients = {df["ikn"].nunique()}')
df = df[df['baseline_albumin_count'].notnull()]
print(f'Size of test data with albumin count = {len(df)}, Number of patients = {df["ikn"].nunique()}')
df = df[df['days_since_starting_chemo'] == 0] # very first treatment
print(f'Size of test data with only first day chemos = {len(df)}, Number of patients = {df["ikn"].nunique()}')


# In[145]:


def compute_score(data):
    data['score'] = 0
    data.loc[data['age'].between(61, 70), 'score'] += 1.5
    data.loc[data['age'] > 70, 'score'] += 2.5
    data.loc[data['baseline_albumin_count'] < 35, 'score'] += 2.0
    data.loc[data['cisplatin_dosage'].between(101, 150), 'score'] += 1.0
    data.loc[data['cisplatin_dosage'] > 150, 'score'] += 3.0
    data.loc[data['hypertension'], 'score'] += 2.0
    data['score'] /= data['score'].max()
    return data['score']


# In[151]:


split = 'Test'
score = compute_score(df)
labels = {split: Y_test.loc[df.index]}
preds = {split: {'ENS': train_ens.preds[split]['ENS'].loc[df.index],
                 'MSB': pd.DataFrame({col: score for col in Y_test.columns})}}
eval_motwani_model = Evaluate(output_path='', preds=preds, labels=labels, orig_data=df)


# In[152]:


# label distribtuion
labels[split].apply(pd.value_counts)


# In[153]:


kwargs = {'algorithms': ['ENS', 'MSB'], 'splits': ['Test'], 'display_ci': True, 'save_score': False}
result = eval_motwani_model.get_evaluation_scores(**kwargs)
result


# In[154]:


eval_motwani_model.all_plots_for_single_target(algorithm='MSB', target_event='AKI', split='Test',
                                               n_bins=20, calib_strategy='quantile', figsize=(12,12), save=False)


# In[155]:


points = np.arange(0, 8.6, 0.5)/8.5 # 8.5 is the highest score possible, 0 is lowest score possible
eval_motwani_model.operating_points('MSB', points, metric='threshold', target_events=['AKI'], split='Test', save=False)


# ### Compare with ENS

# In[156]:


eval_motwani_model.all_plots_for_single_target(algorithm='ENS', target_event='AKI', split='Test',
                                               n_bins=20, calib_strategy='quantile', figsize=(12,12), save=False)


# In[157]:


points = np.arange(0, 8.6, 0.5)/8.5 # 8.5 is the highest score possible, 0 is lowest score possible
eval_motwani_model.operating_points('ENS', points, metric='threshold', target_events=['AKI'], split='Test', save=False)


# ## Missingness By Splits

# In[158]:


from src.utility import get_nmissing_by_splits


# In[159]:


# Acute Kidney Injury
missing = get_nmissing_by_splits(model_data, train_ens.labels)
missing.sort_values(by=(f'Test (N={len(Y_test)})', 'Missing (N)'), ascending=False)


# In[167]:


# Chronic Kidney Disease
missing = get_nmissing_by_splits(model_data, train_ens.labels)
missing.sort_values(by=(f'Test (N={len(Y_test)})', 'Missing (N)'), ascending=False)


# In[ ]:
