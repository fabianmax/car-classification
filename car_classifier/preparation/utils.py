import re
import pandas as pd


def open_file_structure(path, clean=True):
    """
    Open and prepare file structure

    Args:
        path: path to file containing picture naming schema
        clean: If filenames should be cleaned (e.g. to lower cases)

    Returns:
         List cof strings giving the header names
    """

    with open(path) as f:
        structure = f.read()

    structure = [x.replace("'", "") for x in structure.split("', ")]

    if clean:
        # To lower, remove special characters and whitespaces
        structure = [x.lower() for x in structure]
        structure = [re.sub('[^a-zA-Z0-9 \n\.]', '', x) for x in structure]
        structure = [x.replace(' ', '_') for x in structure]
        structure = [x.replace('__', '_') for x in structure]

    return structure


def expand_column(df, column, col_names):
    """
    Extract columns for single string

    Args:
        df: Data.Frame with raw information
        column: Name of column to be expanded into several columns
        col_names: List of names used as new column names

    Returns
        A pandas DataFrame object
    """
    df_expanded = pd.DataFrame(df[column].str.split('_', expand=True))
    df_expanded.columns = [*col_names, 'id']
    return df_expanded