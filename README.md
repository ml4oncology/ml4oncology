# Chemotherapy Adverse Events Detection

ML pipeline for data processing and model training for the prediction of adverse events in cancer patients. Adverse events include
- cytopenia (a condition in which blood count levels are dangerously low) prior to their upcoming chemotherapy sessions
- acute care (emergency department visits and hospitalizations) within a month after their chemotherapy session
- acute kidney injury within 22 days or prior to their upcoming chemotherapy sessions

The dataset, provided by ICES [1], is a population based administrative data consisting of data on demographics, cancer diagnosis and treatment, symptom questionnaires, and lab tests for approximately 120000 patients in Ontario, Canada between 2005 and 2020. By predicting adverse events in chemotherapy patients before their next scheduled chemotherapy administration, these models can aid in early intervention and facilitate tailored treatment.

The full report for cytopenia prediction can be found in "A Warning System for Cytopenias during Chemotherapy using Population-Based Administrative Data.pdf".

This dataset is not open-source. The code can only be run in the ICES DSH (Data Safe Haven) environment.

## Instructions

    python <adverse event folder>/Preprocess.py 
    python <adverse event folder>/Training.py

or convert those python files into jupyter notebooks, and run them on a jupyter browser.

## Prerequisites
See requirements.txt

## References
[1] ICES. Data Discovery Better Health. https://www.ices.on.ca/. Accessed Aug 16, 2021.