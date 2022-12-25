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
Used for simple processing of large dataset using pyspark
"""
import glob
import pickle
import shutil

from difflib import SequenceMatcher
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col, collect_set, expr, length, row_number, substring, to_date, 
    to_timestamp
)
from scipy.stats import mode
import pandas as pd

from src.config import (
    root_path, 
    OBS_CODE, OBS_VALUE, OBS_DATE, OBS_RDATE, 
    olis_cols
)

# Helper functions
def clean_string(df, cols):
    # remove first two characters "b'" and last character "'"
    for col in cols:
        # e.g "b'718-7'" start at the 3rd char (the 7), cut off after 8 (length
        # of string) - 3 = 5 characters. We get "718-7"
        df = df.withColumn(col, expr(f"substring({col}, 3, length({col})-3)"))
    return df

def spark_handler(func):
    def wrapper(*args, **kwargs):
        spark = SparkSession.builder \
            .config("spark.driver.memory", "15G") \
            .appName("Main") \
            .getOrCreate()
        sc = spark.sparkContext
        sc.setLogLevel('ERROR')
        result = func(spark, *args, **kwargs)
        spark.stop()
        return result
    return wrapper

# OLIS (lab test) data 
def filter_olis_data(olis, chemo_ikns, observations=None):
    """
    Args:
        chemo_ikns (set): A sequence of ikns (str) we want to keep
        observations (set): A sequence of observation codes (str) associated 
            with specific lab tests we want to keep. If None, no observations 
            are excluded.
    """
    # organize and format columns
    olis = olis.select(olis_cols)
    olis = clean_string(olis, ['ikn', OBS_CODE, 'ReferenceRange', 'Units'])
    olis = olis.withColumnRenamed(OBS_VALUE, 'value')
    olis = olis.withColumn('value', olis['value'].cast('double')) 
    olis = olis.withColumn(OBS_DATE, to_date(OBS_DATE))
    olis = olis.withColumn(OBS_RDATE, to_timestamp(OBS_RDATE))
    
    # filter patients not in chemo_df
    olis = olis.filter(olis['ikn'].isin(chemo_ikns))

    if observations is not None:
        # filter rows with excluded observations
        olis = olis.filter(olis[OBS_CODE].isin(observations))
        
    # remove rows with blood count null or neg values
    olis = olis.filter(~(olis['value'].isNull() | (olis['value'] < 0)))
    
    # remove duplicate rows
    subset = ['ikn', OBS_CODE, OBS_DATE, 'value']
    olis = olis.dropDuplicates(subset) 
    
    # if only the patient id, blood, and observation timestamp are duplicated 
    # (NOT the blood count value), keep the most recently RELEASED row
    subset = ['ikn', OBS_CODE, OBS_DATE]
    window = Window.partitionBy(*subset).orderBy(col(OBS_RDATE).desc())
    olis = olis.withColumn('row_number', row_number().over(window))
    olis = olis.filter(olis['row_number'] == 1).drop('row_number')
    
    return olis

@spark_handler
def preprocess_olis_data(spark, save_path, chemo_ikns, observations=None):
    olis = spark.read.csv(f'{root_path}/data/olis.csv', header=True)
    olis = filter_olis_data(olis, chemo_ikns, observations)
    olis.coalesce(1).write.csv(f'{save_path}/tmp', header=True)
    # Rename and move the data from the temorary directory created by PySpark, 
    # and remove the temporary directory
    files = glob.glob(f'{save_path}/tmp/part*')
    assert len(files) == 1
    file = files[0]
    shutil.move(file, f'{save_path}/olis.csv')
    shutil.rmtree(f'{save_path}/tmp')

# Extract observation units
def clean_unit(unit):
    unit = unit.lower()
    unit = unit.replace(' of ', '')
    splits = unit.split(' ')
    if splits[-1].startswith('cr'): # e.g. mg/mmol creat
        assert(len(splits) == 2)
        unit = splits[0] # remove the last text
    
    for c in ['"', ' ', '.']: unit = unit.replace(c, '')
    for c in ['-', '^', '*']: unit = unit.replace(c, 'e')
    if ((SequenceMatcher(None, unit, 'x10e9/l').ratio() > 0.5) or 
        (unit == 'bil/l')): 
        unit = 'x10e9/l'
    if unit in {'l/l', 'ratio', 'fract', '%cv'}: 
        unit = '%'
    unit = unit.replace('u/', 'unit/')
    unit = unit.replace('/l', '/L')
    return unit

@spark_handler
def extract_observation_units(spark):
    olis = spark.read.csv(f'{root_path}/data/olis.csv', header=True)
    olis = clean_string(olis, [OBS_CODE, 'Units'])
    observation_units = olis.groupBy(OBS_CODE).agg(collect_set('Units'))
    observation_units = observation_units.toPandas()
    observation_units = dict(observation_units.values)
    
    unit_map = {}
    for obs_code, units in observation_units.items():
        units = [clean_unit(unit) for unit in units]
        # WARNING: there is a possibility the most frequent unit may be the 
        # wrong unit. Not enough manpower to check each one manually
        unit_map[obs_code] = mode(units)[0][0]
        
    filename = f'{root_path}/data/olis_units.pkl'
    with open(filename, 'wb') as file:    
        pickle.dump(unit_map, file)
        
    return unit_map
