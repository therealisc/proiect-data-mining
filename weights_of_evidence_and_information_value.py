import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype


# t - tabelul cu variabilele prelucrate, predictori si tinta
# target - numele variabilei tinta
# bins - numar intervale discretizare

def calculate(data, predictori, tinta, bins=10):
    # Creare tabele
    information_value, weights_of_evidence = pd.DataFrame(), pd.DataFrame()
    # Selectia predictorilor
    for predictor in predictori:
        if is_numeric_dtype(data[predictor]) and (len(np.unique(data[predictor])) > 10):
            binned_x = pd.qcut(data[predictor], bins, duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[tinta]})
        else:
            d0 = pd.DataFrame({'x': data[predictor], 'y': data[tinta]})

        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        # print("--->\n",d)
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        # d['% of Events'] = d['Events'] / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()

        # d['% of Non-Events'] = d['Non-Events'] / d['Non-Events'].sum()
        # d['WoE'] = np.log(d['% of Events'] / d['% of Non-Events'])
        d['WoE'] = np.log(d['% of Events']) - np.log(d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        # print("--->\n",d)
        d.insert(loc=0, column='Variabile', value=predictor)
        temp = pd.DataFrame({"Variabile": [predictor], "IV": [
                            d['IV'].sum()]}, columns=["Variabile", "IV"])
        information_value = pd.concat([information_value, temp], axis=0)
        weights_of_evidence = pd.concat([weights_of_evidence, d], axis=0)

    return information_value, weights_of_evidence
