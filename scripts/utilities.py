"""
========================================================================
© 2021 Institute for Clinical Evaluative Sciences. All rights reserved.

TERMS OF USE:
##Not for distribution.## This code and data is provided to the user solely for its own non-commercial use by individuals and/or not-for-profit corporations. User shall not distribute without express written permission from the Institute for Clinical Evaluative Sciences.

##Not-for-profit.## This code and data may not be used in connection with profit generating activities.

##No liability.## The Institute for Clinical Evaluative Sciences makes no warranty or representation regarding the fitness, quality or reliability of this code and data.

##No Support.## The Institute for Clinical Evaluative Sciences will not provide any technological, educational or informational support in connection with the use of this code and data.

##Warning.## By receiving this code and data, user accepts these terms, and uses the code and data, solely at its own risk.
========================================================================
"""
""" Utilities for data analysis
"""
import tqdm
import pandas as pd
import numpy as np
import multiprocessing as mp
import datetime as dt
import matplotlib.pyplot as plt

from collections import Counter

def sas_to_csv(name):
    """Convert sas to csv
    """
    def create_clean_regimen_column(df):
        df['regimen'] = df['cco_regimen'].astype(str)
        df['regimen'] = df['regimen'].str[2:-1]
        df.loc[df['regimen'].str.contains('NCT'), 'regimen'] = 'TRIALS'
        df['regimen'] = df['regimen'].str.replace("*", "")
        df['regimen'] = df['regimen'].str.replace(" ", "")
        df['regimen'] = df['regimen'].str.lower()
        return df

    chunks = pd.read_sas(f'ykaliwal/{name}.sas7bdat', chunksize=10**6)
    for i, chunk in tqdm.tqdm(enumerate(chunks)):
        # remove empty rows
        chunk = chunk[~chunk.iloc[:,1:].isnull().all(axis=1)]

        if name == 'olis':
            # keep only ObservationCode 777-3 (PLATELETS), 751-8 (NEUTROPHIL), 718-7 (HEMOGLOBIN)
            chunk = chunk[chunk['ObservationCode'].isin([b'777-3',b'751-8',b'718-7'])]

        if name == 'systemic':
            # create cleaned up regimen column
            chunk = create_clean_regimen_column(chunk.copy())
    
        # write to csv
        header = True if i == 0 else False
        chunk.to_csv(f"data/{name}.csv", header=header, mode='a', index=False)

def num_regimen_occurence():
    """Get most occuring chemotherapy regimens based on
        1. number of rows (inlcudes all the different drugs)
        2. number of chemo sessions
        3. number of patients
    """
    regimen_count1 = Counter() # based on number of rows (inlcudes all the different drug counts)
    regimen_count2 = Counter() # based on chemo sessions
    regimen_count3 = {} # based on number of patients

    chunks = pd.read_csv('data/systemic1.csv', chunksize=10**6) # (chunksize=10**5, i=142, 00:32), (chunksize=10**6, i=14, 00:32)
    for i, chunk in tqdm.tqdm(enumerate(chunks)):
        # remove first two characters "b'" and last character "'"
        chunk['ikn'] = chunk['ikn'].str[2:-1]
        
        # update occurence of regiments based on number of rows
        regimen_count1 += Counter(dict(chunk['regimen'].value_counts()))
        
        # keep only target columns
        chunk = chunk[['ikn', 'regimen', 'visit_date']]
        chunk = chunk.drop_duplicates()
        # update occurence of regiments based on number of chemo regiments
        regimen_count2 += Counter(dict(chunk['regimen'].value_counts()))
        
        # update occurence of regiments based on number of patients
        for regimen, group in chunk.groupby('regimen'):
            if regimen not in regimen_count3:
                regimen_count3[regimen] = set(group['ikn'].unique())
            else:
                regimen_count3[regimen].update(group['ikn'].unique())

    regimen_count3 = {regimen:len(ikns) for regimen, ikns in regimen_count3.items()}
    regimen_count3 = sorted(regimen_count3.items(), key=lambda x: x[1], reverse=True)

    return regimen_count1, regimen_count2, regimen_count3

def get_blood_ranges():
    """Get reference ranges for each blood type
    """
    blood_mapping = {'777-3': 'platelet', '751-8':'neutrophil', '718-7':'hemoglobin'}
    blood_ranges = {blood_type: [np.inf, 0] for blood_type in blood_mapping.values()}
    cols = ['ReferenceRange', 'ObservationCode']
    chunks = pd.read_csv("data/olis1.csv", chunksize=10**6) # (chunksize=10**5, i=653, 01:45), (chunksize=10**6, i=66, 1:45)
    for i, chunk in tqdm.tqdm(enumerate(chunks)):
        # keep columns of interest
        chunk = chunk[cols].copy()
        # remove rows where no reference range is given
        chunk = chunk[~chunk['ReferenceRange'].isnull()]
        # remove first two characters "b'" and last character "'"
        for col in cols:
            chunk[col] = chunk[col].str[2:-1]
        # map the blood type to blood code
        chunk['ObservationCode'] = chunk['ObservationCode'].map(blood_mapping)
        # get the min/max blood count values for this chunk and update the global min/max blood range
        for blood_type, group in chunk.groupby('ObservationCode'):
            ranges = group['ReferenceRange'].str.split('-')
            min_count = min(ranges.str[0].replace(r'^\s*$', np.nan, regex=True).fillna('inf').astype(float))
            max_count = max(ranges.str[1].replace(r'^\s*$', np.nan, regex=True).fillna('0').astype(float))
            blood_ranges[blood_type][0] = min(min_count, blood_ranges[blood_type][0])
            blood_ranges[blood_type][1] = max(max_count, blood_ranges[blood_type][1])
    return blood_ranges

def read_partially_reviewed_csv():
    """Extract the contents from regiments.csv that contains chemo cycle durations
    """
    df = open('DATAPATH/regiments.csv')
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
    """Filter through regiments.csv to only keep included regimens
    """
    df = df[df['include (1) or exclude (0)']=='1'] 
    df = df.drop(columns=['include (1) or exclude (0)'])
    df = df.set_index('regiments')
    return df

class Plots:
    def __init__(self):
        pass

    def plot_regimen_count(self):
        """Plot the top 20 most common regimen count based on 
            1. number of rows
            2. number of chemo sessions
            3. number of patients
        """
        regimen_count1, regimen_count2, regimen_count3 = num_regimen_occurence()

        regimens, count = zip(*regimen_count1.most_common(n=20))
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(1,3,1)
        plt.title('Top 20 chemo regimen occurence\n based on number of rows')
        plt.bar(regimens, count)
        plt.xticks(rotation=90)

        regimens, count = zip(*regimen_count2.most_common(n=20))
        ax = fig.add_subplot(1,3,2)
        plt.title('Top 20 chemo regimen occurence\n based on number of chemo sessions')
        plt.bar(regimens, count)
        plt.xticks(rotation=90)

        regimens, count = zip(*regimen_count3[0:20])
        ax = fig.add_subplot(1,3,3)
        plt.title('Top 20 chemo regimen occurence\n based on number of patients')
        plt.bar(regimens, count)
        plt.xticks(rotation=90)

        plt.show()

    def plot_num_patients_per_regimen(self, df):
        """Plot number of patients per cancer regiment
        """
        regiments = []
        patients = []
        for regimen, group in df.groupby('regimen'):
            regiments.append(regimen)
            patients.append(len(group['ikn'].unique()))
        plt.bar(regiments, patients) 
        plt.xlabel('Chemotherapy Regiments')
        plt.ylabel('Number of Patients')
        plt.xticks(rotation=90)

    def plot_num_blood_counts_per_regimen(self, df):
        """Plot number of blood measurements per cancer regiment
        """
        regiments = []
        blood_counts = []
        for regimen, group in df.groupby('regimen'):
            regiments.append(regimen)
            blood_counts.append(sum((~group[range(-5,29)].isnull()).sum(axis=1)))
        plt.bar(regiments, blood_counts) 
        plt.xlabel('Chemotherapy Regiments')
        plt.ylabel('Number of Blood Measurements')
        plt.xticks(rotation=90)

    def hist_blood_counts(self, df):
        """plot histogram of numbers of blood counts measured (for a single chemo session)
        """
        df = df[range(-5,29)]
        blood_counts = (~df[range(-5,29)].isnull()).sum(axis=1).values
        n, bins, patches = plt.hist(blood_counts, bins=33)
        plt.title('Histogram of Number of Blood Measurements for a Single Session')

    def scatter_plot(self, df, cycle_lengths, unit='10^9/L', save=False, filename="NEUTROPHIL_PLOT1"):
        """Create scatter plot of average blood count values over a chemo cycle for each regimen
        """
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
            plt.savefig(f'plots/{filename}.jpg')
        plt.show()

    def below_threshold_bar_plot(self, df, cycle_lengths, threshold, save=False, 
                                    filename='NEUTROPHIL_PLOT2', color=None):
        """Create bar plot of number of patients with blood count below dangerous threshold over a chemo cycle
        for each regimen
        """
        num_regimen = len(set(df['regimen']))
        fig = plt.figure(figsize=(15, 150))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        idx = 1
        for regimen, group in tqdm.tqdm(df.groupby('regimen')):
            # print(regimen)
            # print("Ratio of blood count versus nans (how sparse the dataframe is) =", (~group[range(-5,28)].isnull()).values.sum() / (len(group)*33))
            cycle_length = int(cycle_lengths[regimen])
            num_patients = (~group[range(0,cycle_length+1)].isnull()).sum(axis=0).values
            # cannot display data summary with observations less than 6, replace them with 5
            num_patients[(num_patients < 6) & (num_patients > 0)] = 6
            ax = fig.add_subplot(num_regimen,3,idx)
            plt.bar(range(0,cycle_length+1), num_patients, color=color)
            plt.title(regimen)
            plt.ylabel('Number of patients')
            plt.xlabel('Day')
            if cycle_length < 15: plt.xticks(range(0, cycle_length+1))
      
            num_patient_below_threshold = np.array([len(group.loc[group[day] < threshold, 'ikn'].unique()) for day in range(0,cycle_length+1)])
            # cannot display data summary with observations less than 6, replace them with 5
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
            plt.savefig(f'plots/{filename}.jpg', bbox_inches='tight')
        plt.show()

    def iqr_plot(self, df, cycle_lengths, unit='10^9/L', show_outliers=True, save=False, 
                 filename='NEUTROPHIL_PLOT3', figsize=(15,150)):
        """Create iqr plot of blood count values over a chemo cycle for each regimen
        """
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
            plt.savefig(f'plots/{filename}.jpg', bbox_inches='tight')    
        plt.show()
        nadir_df = pd.DataFrame(nadir_dict)
        return nadir_df.T

    def mean_cycle_plot(self, df, cycle_lengths, unit='10^9/L', save=False, filename='NEUTROPHIL_PLOT4'):
        """Plot average blood count values over a chemo cycle for the first five cycles for each regimen
        """
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
            plt.savefig(f'plots/{filename}.jpg')
        plt.show()