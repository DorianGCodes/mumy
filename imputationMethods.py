import pandas as pd

def fillWithMean(dataframe):
    return dataframe.fillna(dataframe.mean())
