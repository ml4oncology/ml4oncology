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
import pandas as pd
from scripts.utility import get_eGFR

def test_eGFR():
    # kidney.org/professionals/kdoqi/gfr_calculator
    test_df = pd.DataFrame([[70, 200, 'M'],
                            [70, 200, 'F'],
                            [70, 100, 'M'],
                            [70, 100, 'F']], columns=['age', 'value', 'sex'])
    answer = [30, 23, 70, 52]
    test_df = get_eGFR(test_df)
    assert all(test_df['eGFR'].round() == answer)
