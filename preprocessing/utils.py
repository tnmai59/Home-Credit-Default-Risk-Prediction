"""
This file contains all the ultility functions that are neccessary for other pipelines in this project
"""
import os
import gc
import time
import numpy as np
import pandas as pd
from contextlib import contextmanager
import multiprocessing as mp
from functools import partial
from scipy.stats import kurtosis, iqr, skew
from configurations import NUM_THREADS
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

def group(df_to_agg, prefix, aggregations, aggregate_by= 'SK_ID_CURR'):
    """
    Perform aggregation on a DataFrame grouped by a specified column.

    Parameters:
    - df_to_agg: The DataFrame to be aggregated.
    - prefix: A string prefix to be added to the names of the aggregated columns.
    - aggregations: A dictionary specifying the aggregation functions for each column.
    - aggregate_by: The column by which the DataFrame should be grouped. Default is 'SK_ID_CURR'.

    Returns:
    - agg_df: The aggregated DataFrame.

    Example Usage:
    agg_df = group(df, 'app_', {'AMT_INCOME_TOTAL': 'mean', 'AMT_CREDIT': 'max'})
    """
    agg_df = df_to_agg.groupby(aggregate_by).agg(aggregations)
    agg_df.columns = pd.Index(['{}{}_{}'.format(prefix, e[0], e[1].upper())
                               for e in agg_df.columns.tolist()])
    return agg_df.reset_index()

def group_and_merge(df_to_agg, df_to_merge, prefix, aggregations, aggregate_by= 'SK_ID_CURR'):
    """
    Perform aggregation on one DataFrame and merge the aggregated results into another DataFrame.

    Parameters:
    - df_to_agg: The DataFrame to be aggregated.
    - df_to_merge: The DataFrame to merge the aggregated results into.
    - prefix: A string prefix to be added to the names of the aggregated columns.
    - aggregations: A dictionary specifying the aggregation functions for each column.
    - aggregate_by: The column by which the DataFrame should be grouped. Default is 'SK_ID_CURR'.

    Returns:
    - merged_df: The DataFrame resulting from the merge.

    Example Usage:
    merged_df = group_and_merge(df1, df2, 'app_', {'AMT_INCOME_TOTAL': 'mean', 'AMT_CREDIT': 'max'})
    """
    agg_df = group(df_to_agg, prefix, aggregations, aggregate_by= aggregate_by)
    return df_to_merge.merge(agg_df, how='left', on= aggregate_by)


def mean_agg(df, group_cols, counted, agg_name):
    """
    Perform mean aggregation on a DataFrame and merge the result back into the original DataFrame.

    Parameters:
    - df: The original DataFrame.
    - group_cols: List of column names to group by.
    - counted: The column name for which mean is calculated.
    - agg_name: The name to be given to the new aggregated column.

    Returns:
    - df: The DataFrame with the mean aggregated result merged back.

    Example Usage:
    df = mean_agg(df, ['group_column1', 'group_column2'], 'counted_column', 'mean_counted_column')
    """
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].mean().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def median_agg(df, group_cols, counted, agg_name):
    """
    Perform median aggregation on a DataFrame and merge the result back into the original DataFrame.

    Parameters:
    - df: The original DataFrame.
    - group_cols: List of column names to group by.
    - counted: The column name for which median is calculated.
    - agg_name: The name to be given to the new aggregated column.

    Returns:
    - df: The DataFrame with the aggregated result merged back.

    Example Usage:
    df = median_agg(df, ['group_column1', 'group_column2'], 'counted_column', 'median_counted_column')
    """
    # Group by specified columns and calculate the median for the specified column
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].median().reset_index().rename(
        columns={counted: agg_name})

    gp = df[group_cols + [counted]].groupby(group_cols)[counted].median().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def std_agg(df, group_cols, counted, agg_name):
    """
    Perform standard deviation aggregation on a DataFrame and merge the result back into the original DataFrame.

    Parameters:
    - df: The original DataFrame.
    - group_cols: List of column names to group by.
    - counted: The column name for which standard deviation is calculated.
    - agg_name: The name to be given to the new aggregated column.

    Returns:
    - df: The DataFrame with the standard deviation aggregated result merged back.

    Example Usage:
    df = std_agg(df, ['group_column1', 'group_column2'], 'counted_column', 'std_counted_column')
    """
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].std().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def sum_agg(df, group_cols, counted, agg_name):
    """
    Perform sum aggregation on a DataFrame and merge the result back into the original DataFrame.

    Parameters:
    - df: The original DataFrame.
    - group_cols: List of column names to group by.
    - counted: The column name for which sum is calculated.
    - agg_name: The name to be given to the new aggregated column.

    Returns:
    - df: The DataFrame with the sum aggregated result merged back.

    Example Usage:
    df = sum_agg(df, ['group_column1', 'group_column2'], 'counted_column', 'sum_counted_column')
    """
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].sum().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def one_hot_encoder(df, categorical_columns=None, nan_as_category=True):
    """
    Perform one-hot encoding on categorical columns in a DataFrame.

    Parameters:
    - df: The DataFrame to be one-hot encoded.
    - categorical_columns: List of column names to be one-hot encoded. If not provided, it selects all object-type columns.
    - nan_as_category: If True, treat NaN values as a separate category.

    Returns:
    - df: The DataFrame with one-hot encoding applied.
    - categorical_columns: List of names of newly created one-hot encoded columns.

    Example Usage:
    df, encoded_columns = one_hot_encoder(df, ['categorical_column1', 'categorical_column2'])
    """
    # Record the original columns in the DataFrame
    original_columns = list(df.columns)
    # if categorical columns is not provided, it will automatically search the fearures wit dtype is object in the dataframe
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    categorical_columns = [c for c in df.columns if c not in original_columns]
    return df, categorical_columns

def add_features(feature_name, aggs, features, feature_names, groupby):
    """
    Add new features to a DataFrame based on specified aggregation functions and grouping.

    Parameters:
    - feature_name: The name of the feature to be aggregated.
    - aggs: List of aggregation functions to be applied.
    - features: The DataFrame to which new features are added.
    - feature_names: List of existing feature names.
    - groupby: The DataFrame used for grouping.

    Returns:
    - features: The DataFrame with new features added.
    - feature_names: The updated list of feature names.

    Example Usage:
    features, feature_names = add_features('column_name', ['mean', 'max'], existing_features, existing_feature_names, groupby_df)
    """
    # Extend the list of feature names with new names based on aggregation functions
    feature_names.extend(['{}_{}'.format(feature_name, agg) for agg in aggs])
    for agg in aggs:
        if agg == 'kurt':
            agg_func = kurtosis
        elif agg == 'iqr':
            agg_func = iqr
        else:
            agg_func = agg

        g = groupby[feature_name].agg(agg_func).reset_index().rename(index=str,
                                                                     columns={feature_name: '{}_{}'.format(feature_name,agg)})
        features = features.merge(g, on='SK_ID_CURR', how='left')
    return features, feature_names

def add_features_in_group(features, gr_, feature_name, aggs, prefix):
    """
    Add aggregated features to a DataFrame based on specified aggregation functions and grouping.

    Parameters:
    - features: The DataFrame to which new features are added.
    - gr_: The DataFrame used for grouping.
    - feature_name: The name of the feature to be aggregated.
    - aggs: List of aggregation functions to be applied.
    - prefix: The prefix to be added to the names of the new aggregated features.

    Returns:
    - features: The DataFrame with new aggregated features added.

    Example Usage:
    features = add_features_in_group(features, groupby_df, 'column_name', ['mean', 'max'], 'prefix')
    """
    for agg in aggs:
        if agg == 'sum':
            features['{}{}_sum'.format(prefix, feature_name)] = gr_[feature_name].sum()
        elif agg == 'mean':
            features['{}{}_mean'.format(prefix, feature_name)] = gr_[feature_name].mean()
        elif agg == 'max':
            features['{}{}_max'.format(prefix, feature_name)] = gr_[feature_name].max()
        elif agg == 'min':
            features['{}{}_min'.format(prefix, feature_name)] = gr_[feature_name].min()
        elif agg == 'std':
            features['{}{}_std'.format(prefix, feature_name)] = gr_[feature_name].std()
        elif agg == 'count':
            features['{}{}_count'.format(prefix, feature_name)] = gr_[feature_name].count()
        elif agg == 'skew':
            features['{}{}_skew'.format(prefix, feature_name)] = skew(gr_[feature_name])
        elif agg == 'kurt':
            features['{}{}_kurt'.format(prefix, feature_name)] = kurtosis(gr_[feature_name])
        elif agg == 'iqr':
            features['{}{}_iqr'.format(prefix, feature_name)] = iqr(gr_[feature_name])
        elif agg == 'median':
            features['{}{}_median'.format(prefix, feature_name)] = gr_[feature_name].median()
    return features

def parallel_apply(groups, func, index_name='Index', num_workers=0, chunk_size=100000):
    """
    Apply a function to groups in parallel using multiprocessing.

    Parameters:
    - groups: A pandas GroupBy object.
    - func: The function to be applied to each group.
    - index_name: The name to be given to the index of the resulting DataFrame.
    - num_workers: The number of parallel workers to use. If not specified (0), it defaults to NUM_THREADS.
    - chunk_size: The size of each chunk of groups to be processed in parallel.

    Returns:
    - A DataFrame with the results of applying the function to groups.

    Example Usage:
    result_df = parallel_apply(groups=grouped_data, func=my_function, index_name='GroupIndex', num_workers=4, chunk_size=50000)
    """
    # Check if the number of workers is not specified or is less than or equal to 0
    if num_workers <= 0:
        # Set the number of workers to the default value (NUM_THREADS)
        num_workers = NUM_THREADS

    # Initialize empty lists to store index and features
    indeces, features = [], []

    # Iterate over chunks of groups using the chunk_groups function
    for index_chunk, groups_chunk in chunk_groups(groups, chunk_size):
        # Use multiprocessing to parallelize the application of the function to groups
        with mp.pool.Pool(num_workers) as executor:
            # Apply the function to each group in parallel
            features_chunk = executor.map(func, groups_chunk)

        # Extend the lists with the results from the parallel processing
        features.extend(features_chunk)
        indeces.extend(index_chunk)

    # Create a DataFrame from the collected features
    features = pd.DataFrame(features)

    # Set the index and index name of the resulting DataFrame
    features.index = indeces
    features.index.name = index_name

    # Return the DataFrame with the applied function results
    return features

def chunk_groups(groupby_object, chunk_size):
    """
    Yield chunks of groups from a pandas GroupBy object.

    Parameters:
    - groupby_object: The pandas GroupBy object to be chunked.
    - chunk_size: The size of each chunk.

    Yields:
    - Tuple of index chunks and corresponding data chunks from the GroupBy object.

    Example Usage:
    for index_chunk, data_chunk in chunk_groups(grouped_data, 50000):
        # Process each chunk of groups
        process_chunk(index_chunk, data_chunk)
    """
    # Get the total number of groups in the GroupBy object
    n_groups = groupby_object.ngroups

    # Initialize empty lists to store chunks of groups and corresponding indices
    group_chunk, index_chunk = [], []

    # Iterate over the groups in the GroupBy object
    for i, (index, df) in enumerate(groupby_object):
        # Append the data (group) and index to the lists
        group_chunk.append(df)
        index_chunk.append(index)

        # Check if the chunk size is reached or if it's the last group
        if (i + 1) % chunk_size == 0 or i + 1 == n_groups:
            # Create copies of the lists to yield and reset the lists
            group_chunk_, index_chunk_ = group_chunk.copy(), index_chunk.copy()
            group_chunk, index_chunk = [], []

            # Yield the chunks as a tuple
            yield index_chunk_, group_chunk_

def reduce_memory(df):
    """
    Reduce memory usage of a dataframe by setting data types.

    Parameters:
    - df: The pandas DataFrame to be optimized.

    Returns:
    - The optimized DataFrame with reduced memory usage.
    """
    # Get the initial memory usage of the DataFrame
    start_mem = df.memory_usage().sum() / 1024 ** 2

    # Print initial memory usage information
    print('Initial df memory usage is {:.2f} MB for {} columns'
          .format(start_mem, len(df.columns)))

    # Iterate over columns to optimize data types and reduce memory usage
    for col in df.columns:
        # Get the data type of the column
        col_type = df[col].dtypes

        # Check if the data type is not 'object' (i.e., non-string)
        if col_type != object:
            # Get the minimum and maximum values of the column
            cmin = df[col].min()
            cmax = df[col].max()

            # Optimize integer data types
            if str(col_type)[:3] == 'int':
                # Can use unsigned int here too
                if cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif cmin > np.iinfo(np.int64).min and cmax < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:  # Optimize float data types
                if cmin > np.finfo(np.float16).min and cmax < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    # Get the final memory usage of the optimized DataFrame
    end_mem = df.memory_usage().sum() / 1024 ** 2

    # Calculate the percentage reduction in memory usage
    memory_reduction = 100 * (start_mem - end_mem) / start_mem

    # Print final memory usage information
    print('Final memory usage is: {:.2f} MB - decreased by {:.1f}%'.format(end_mem, memory_reduction))

    # Return the optimized DataFrame with reduced memory usage
    return df

