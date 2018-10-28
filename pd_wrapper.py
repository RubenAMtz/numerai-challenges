"""
Module as a helper for pandas
"""

import pandas as pd
import numpy as np

def dim(dataframe):
    return dataframe.shape

def get_columns(dataframe, columns):
    """
    Returns the columns of the dataframe
    """
    return dataframe.loc[:, columns]

def get_rows(dataframe, value, column):
    """
    Returns the rows specified of the dataframe in the column
    """
    return dataframe.loc[ dataframe[column] == value ]

def k_fold_by_attribute(dataframe, column_name):
    """
    Returns a dictionary with the dataframe associated to every attribute in column
    """
    # get unique values from column
    target_column = dataframe[column_name]
    target_uniques = target_column.unique()

    # get dataframes from unique values and create a dictionary
    k = { value: get_rows(dataframe, value, column_name) for value in target_uniques }
    return k

def insert_column(dataframe, column, name):
    """
    Inserts a column to the end of the dataframe
    """
    df_size = dim(dataframe)
    last_column = df_size[1]
    dataframe.insert(last_column, name, column)

def test():
    df = pd.DataFrame({'col1':[5, 2, 3, 6], 'col2':[ 9, 2, 4, 7]})
    #print(df)
    #print(dim(df))
    #print(get_columns(df, ['a']))
    insert_column(df, list('abca'), 'col3')
    #print(df)
    #print(get_rows(df, 'b', 'col3'))
    k = k_fold_by_attribute(df, 'col3')
    print(df)
    print('k[a]:')
    print(k['a'])
    print('k[b]:')
    print(k['b'])
    print('k[c]:')
    print(k['c'])

if __name__ == '__main__':
    test()
