import pandas as pd
from utils import *
from imputationMethods import *

files = ['kc_hous_6%','kc_hous_15%','kc_hous_20%','kc_hous_40%']
dataframes = {}

for i in files:
    dataframes[i] = (pd.read_csv(filepath_or_buffer='dataSets/'+i,sep=',',header=0,na_values='?'))

for i in files:
    columns_with_nans = dataframes[i][['sqft_living','price']]
    print_statistics(columns_with_nans)
    filled_data_frame = fillWithMean(columns_with_nans) #<- tutaj własny filling trzeba by dać

    print_regression_statistics(drop_nans(columns_with_nans), 'No missing values', i + '-plot-no-nans')
    print_regression_statistics(filled_data_frame, 'Mean', i + '-plot')

    save_missingStats(columns_with_nans,i)
    save_stats_to_file(drop_nans(columns_with_nans), 'No missing values', i + '-plot-no-nans')
    save_stats_to_file(filled_data_frame, 'Mean', i + '-plot')


