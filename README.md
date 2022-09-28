# Chemotherapy Adverse Events Detection

ML pipeline for data processing and model training for the prediction of adverse events in cancer patients. Adverse events include
- cytopenia (a condition in which blood count levels are dangerously low) prior to their upcoming chemotherapy sessions
- acute care use (emergency department visits and hospitalizations) within a month after their chemotherapy session
- nephrotoxicity (acute kidney injurt and chronic kidney disease) within 42 days or prior to their upcoming chemotherapy sessions
- death within weeks, months, and a year after their chemotherapy session

The dataset, provided by ICES [1], is a population based administrative data consisting of data on demographics, cancer diagnosis and treatment, symptom questionnaires, and lab tests for approximately 120000 patients in Ontario, Canada between 2005 and 2020. By predicting adverse events in chemotherapy patients before their next scheduled chemotherapy administration, these models can aid in early intervention and facilitate tailored treatment.

This dataset is not open-source. The code can only be run in the ICES DSH (Data Safe Haven) environment.

## Instructions

    python <adverse event folder>/Preprocess.py 
    python <adverse event folder>/Training.py

or convert those python files into jupyter notebooks, and run them on a jupyter browser.

## Prerequisites
See environment.yaml, or run

	conda env create -f environment.yaml

To use that conda environment on jupyter notebook, run

	python -m ipykernel install --user --name=myenv

## References
[1] ICES. Data Discovery Better Health. https://www.ices.on.ca/. Accessed Aug 16, 2021.