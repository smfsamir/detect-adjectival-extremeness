"""Utility functions for applying to pandas dataframes because messing with pandas dfs fuckin sucks
"""
def modify_column_of_df(df, column, f):
    """Modify column of df by applying f to all of the values. 

    Params:
        df {DataFrame}
        column {str}
        f {Any -> Any} -- Function to apply on column 
    Returns:
        {DataFrame} -- The dataframe with the modified column 
    """
    df[column] = df[column].apply(f)
    return df