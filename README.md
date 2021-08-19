# Cytopenia Detection

ML pipeline for data processing and model training for the prediction of cytopenia (a condition in which blood count levels are dangerously low) in cancer patients prior to their upcoming chemotherapy sessions. The dataset, provided by ICES [1], is a population based administrative data consisting of data on demographics and cancer diagnosis and treatment, symptom questionnaires, and blood work for 32,567 patients in Ontario, Canada between 2007 and 2020. By predicting cytopenias in chemotherapy patients before their next scheduled chemotherapy administration, these models can aid in early intervention and facilitate tailored treatment.

The full report can be found in "A Warning System for Cytopenias during Chemotherapy using Population-Based Administrative Data.pdf".

This dataset is not open-source. The code can only be run in the ICES DSH (Data Safe Haven) environment.

## Instructions

    python preprocess.py --[OPTIONS] 
    python train.py --[OPTIONS] 

## Prerequisites
See requirements.txt

## References
[1] ICES. Data Discovery Better Health. https://www.ices.on.ca/. Accessed Aug 16, 2021.