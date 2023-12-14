import time
import numpy as np
import pandas as pd
from contextlib import contextmanager

@contextmanager
def timer(name):
    """
    Context manager decorator to measure the execution time of a specific code block.

    Parameters:
    - name: A string representing the name or description of the code block.

    Usage Example:
    with timer("Some Operation"):
        # Code block to measure execution time for
        some_operation()

    Output:
    "Some Operation - done in [time]s"
    """

    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(name, time.time() - t0))

# Functions to handling outliers
def handling_outliers_3sigma(col, dataset, dataset_apl):
    """
    Handle outliers in a column using the 3-sigma rule.

    Parameters:
    - col: The column name for which outliers are to be handled.
    - dataset: The original dataset containing the column.
    - dataset_apl: The dataset where outliers will be replaced.

    Returns:
    - A list containing the values of the column in the dataset_apl after handling outliers.
    """
    # Get the values of the specified column from the original dataset
    xs = dataset[col]

    # Calculate mean and standard deviation
    mu = xs.mean()
    sigma = xs.std()

    # Define low and high thresholds based on the 3-sigma rule
    low =  mu - 3*sigma
    high = mu + 3*sigma

    # Define a function to handle outlier values
    def _value(x):
        if x < low:
            return low
        elif x > high:
            return high
        else:
            return x

    # Get the values of the specified column from the dataset_apl
    xapl = dataset_apl[col]

    # Map the _value function to each element in xapl to handle outliers
    xnew = list(map(lambda x: _value(x), xapl))

    # Count the number of low and high outlier values in the new dataset
    n_low = len([i for i in xnew if i == low])
    n_high = len([i for i in xnew if i == high])

    # Get the total number of values in the new dataset
    n = len(xapl)

    # Print information about the percentage of low and high outliers
    print('Percentage of low: {:.2f}{}'.format(100 * n_low / n, '%'))
    print('Percentage of high: {:.2f}{}'.format(100 * n_high / n, '%'))
    print('Low value: {:.2f}'.format(low))
    print('High value: {:.2f}'.format(high))
    # Return the list of values after handling outliers
    return xnew

def _count_unique(x):
    """
    Count the number of unique values in a Pandas Series.

    Parameters:
    - x: Pandas Series for which the number of unique values is to be counted.

    Returns:
    - The number of unique values in the given Pandas Series.
    """
    return pd.Series.nunique(x)

def find_features(df):
    """
    Find features in a DataFrame based on the number of unique values.

    Parameters:
    - df: The Pandas DataFrame for which features are to be found.

    Returns:
    - A list of feature column names that have more than 500 unique values (excluding 'SK_ID_CURR').
    """
    # Apply the _count_unique function to count unique values for each column
    tbl_dis_val = df.apply(_count_unique).sort_values(ascending=False)

    # Select columns with more than 500 unique values (excluding 'SK_ID_CURR')
    cols_3sigma = tbl_dis_val[tbl_dis_val > 500].index.tolist()
    cols_3sigma = [c for c in cols_3sigma if c != 'SK_ID_CURR']

    # Return the list of selected feature column names
    return cols_3sigma
