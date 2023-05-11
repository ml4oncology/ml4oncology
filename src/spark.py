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
from collections import Counter
import glob
import pickle

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col, collect_set, expr, length, row_number, substring, to_date, 
    to_timestamp
)
import pandas as pd

from src.config import (
    root_path, 
    all_observations, olis_cols,
    OBS_CODE, OBS_VALUE, OBS_DATE, OBS_RDATE, 
)
from src.utility import clean_unit

def start_spark():
    spark = SparkSession.builder \
        .config("spark.driver.memory", "15G") \
        .appName("Main") \
        .getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')
    return spark

###############################################################################
# Laboratory Test Data
###############################################################################
def filter_lab_data(lab, chemo_ikns, observations=None):
    """
    Args:
        chemo_ikns (set): A sequence of ikns (str) we want to keep
        observations (set): A sequence of observation codes (str) associated 
            with specific lab tests we want to keep. If None, no observations 
            are excluded.
    """
    # organize and format columns
    lab = lab.select(olis_cols)
    lab = lab.withColumnRenamed(OBS_VALUE, 'value')
    lab = lab.withColumn('value', lab['value'].cast('double')) 
    lab = lab.withColumn(OBS_DATE, to_date(OBS_DATE))
    lab = lab.withColumn(OBS_RDATE, to_timestamp(OBS_RDATE))
    
    # filter patients not in chemo_df
    lab = lab.filter(lab['ikn'].isin(chemo_ikns))

    if observations is not None:
        # filter rows with excluded observations
        lab = lab.filter(lab[OBS_CODE].isin(observations))
        
    # remove rows with null or neg lab test values
    lab = lab.filter(~(lab['value'].isNull() | (lab['value'] < 0)))
    
    # remove duplicate rows
    subset = ['ikn', OBS_CODE, OBS_DATE, 'value']
    lab = lab.dropDuplicates(subset) 
    
    # if only the patient id, blood, and observation timestamp are duplicated 
    # (NOT the blood count value), keep the most recently RELEASED row
    subset = ['ikn', OBS_CODE, OBS_DATE]
    window = Window.partitionBy(*subset).orderBy(col(OBS_RDATE).desc())
    lab = lab.withColumn('row_number', row_number().over(window))
    lab = lab.filter(lab['row_number'] == 1).drop('row_number')
    
    return lab

def extract_observation_units(spark):
    lab = spark.read.parquet(f'{root_path}/data/olis', header=True)
    obs_units = lab.groupBy(OBS_CODE).agg(collect_set('Units'))
    obs_units = obs_units.toPandas()
    obs_units = dict(obs_units.to_numpy())

    unit_map = {}
    for obs_code, units in obs_units.items():
        units = [clean_unit(unit) for unit in units]
        # WARNING: there is a possibility the most frequent unit may be the 
        # wrong unit. Not enough manpower to check each one manually
        unit_map[obs_code] = Counter(units).most_common(1)[0][0]
        
    filename = f'{root_path}/data/olis_units.pkl'
    with open(filename, 'wb') as file:    
        pickle.dump(unit_map, file)
        
    return unit_map
