import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def create_nans2(dateframe,nan_percentage = 100.0):
    # dateframe['sqft_living15'] = dateframe['sqft_living15'].sample(frac=0.1)
    columnsName = dateframe.columns.tolist()
    for i in columnsName:
        dateframe[i] = dateframe[i].sample(frac = nan_percentage/100)
    return dateframe

def drop_nans(dataframe):
    ret = dataframe.dropna()
    return ret.reset_index(drop=True)

def print_statistics(dataframe):
    for label, data in dataframe.items():
        print('\n' + label)
        percent_missing = data.isnull().sum() * 100 / len(dataframe)
        print('  %% missing:\t%s' % round(percent_missing, 6))
        print('  Variance: \t%s' % round(data.var(), 6))

def save_missingStats(dataframe,filename):
    f = open('missingStats/' + filename + ".txt", "w")
    for label, data in dataframe.items():
        f.write('\n' + label + '\n')
        percent_missing = data.isnull().sum() * 100 / len(dataframe)
        f.write('  %% missing:\t%s' % round(percent_missing, 6))
        f.write('  Variance: \t%s' % round(data.var(), 6))
    f.close()

def print_regression_statistics(dataframe, header, filename, show_plot=False):
    print('\n\n~~~~~~ ' + header + ':')

    first_column_name = dataframe.columns[0]
    second_column_name = dataframe.columns[1]
    X = dataframe[first_column_name].values.reshape(-1, 1)
    Y = dataframe[second_column_name].values.reshape(-1, 1)

    linear_regressor = LinearRegression()
    linear_regressor.fit(X, Y)
    Y_pred = linear_regressor.predict(X)

    for data, label in [X, first_column_name], [Y, second_column_name]:
        # print('\n' + label)
        # print('  Mean:     \t%s' % round(data.mean(), 6))
        # print('  Std. dev.:\t%s' % round(data.std(), 6))
        # print('  Quantile1:\t%s' % round(np.percentile(data, 25), 6))
        # print('  Quantile2:\t%s' % round(np.percentile(data, 50), 6))
        # print('  Quantile3:\t%s' % round(np.percentile(data, 75), 6))
        print('\n' + label)
        print('Mean\tdeviation.\tQ1\tQ2\tQ3')
        print(str(round(data.mean(),2))+'\t'+str(round(data.std(),2))+'\t'+
              str(round(np.percentile(data,25),2))+'\t'+
              str(round(np.percentile(data,50),2))+'\t'+
              str(round(np.percentile(data,75),2)))

    print('\nRegressor coeficient:\t' + str(linear_regressor.coef_))
    print('Regressor intercept:\t' + str(linear_regressor.intercept_))

    plt.scatter(X, Y, color='blue')
    plt.plot(X, Y_pred, color='red')
    plt.xlabel(first_column_name)
    plt.ylabel(second_column_name)

    plt.savefig('graphs/' + filename)

    if show_plot is True:
        plt.show()

    plt.clf()

def save_stats_to_file(dataframe, header, filename, show_plot=False):
    f = open('stats/' + filename + ".txt","w")

    first_column_name = dataframe.columns[0]
    second_column_name = dataframe.columns[1]
    X = dataframe[first_column_name].values.reshape(-1, 1)
    Y = dataframe[second_column_name].values.reshape(-1, 1)

    linear_regressor = LinearRegression()
    linear_regressor.fit(X, Y)
    Y_pred = linear_regressor.predict(X)

    for data, label in [X, first_column_name], [Y, second_column_name]:
        # print('\n' + label)
        # print('  Mean:     \t%s' % round(data.mean(), 6))
        # print('  Std. dev.:\t%s' % round(data.std(), 6))
        # print('  Quantile1:\t%s' % round(np.percentile(data, 25), 6))
        # print('  Quantile2:\t%s' % round(np.percentile(data, 50), 6))
        # print('  Quantile3:\t%s' % round(np.percentile(data, 75), 6))
        f.write('\n' + label + '\n')
        f.write('Mean\tDeviation\tQ1\tQ2\tQ3\n')
        f.write(str(round(data.mean(),2))+'\t'+str(round(data.std(),2))+'\t'+
              str(round(np.percentile(data,25),2))+'\t'+
              str(round(np.percentile(data,50),2))+'\t'+
              str(round(np.percentile(data,75),2)))

    f.write('\nRegressor coeficient:\n' + str(linear_regressor.coef_))
    f.write('\nRegressor intercept:\n' + str(linear_regressor.intercept_))

    plt.scatter(X, Y, color='blue')
    plt.plot(X, Y_pred, color='red')
    plt.xlabel(first_column_name)
    plt.ylabel(second_column_name)

    plt.savefig('graphs/' + filename)

    if show_plot is True:
        plt.show()

    plt.clf()
    f.close()
def save_dataframe_to_file(dataframe, file_name, missing_percent):
    dataframe.to_csv(file_name.split('.',1)[0] + str(missing_percent) + '.' +
                     file_name.split('.',1)[1], index=False, na_rep='?')

def generateNans(dataframe) :
    first_column = dataframe['price']
    second_column = dataframe['sqft_living']

    columns = dataframe[['price', 'sqft_living']]
    columns_with_nans = create_nans2(columns, 97)

    filePath = "kc_hous_6%"
    columns_with_nans.to_csv(filePath)
    df = pd.read_csv(filepath_or_buffer=filePath, sep=',', header=0, na_values='?')

    perc = df.dropna().shape[0] / df.shape[0]
    print(perc)