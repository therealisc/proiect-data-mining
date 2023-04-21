import pandas as pd
from pandas.api.types import is_numeric_dtype


def codificare(data):
    assert isinstance(data, pd.DataFrame)
    for v in list(data):
        if is_numeric_dtype(data[v]):
            if any(data[v].isna()):
                data[v].fillna(data[v].mean(), inplace=True)
        else:
            if any(data[v].isna()):
                data[v].fillna(data[v].mode()[0], inplace=True)
            data[v] = pd.Categorical(data[v]).codes + 1
