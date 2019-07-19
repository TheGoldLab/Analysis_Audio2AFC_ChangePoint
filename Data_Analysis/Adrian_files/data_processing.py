"""
module containing functions for managing data files from experiment
"""
import pandas as pd
import os


data_mapping = pd.read_csv('../../data_mapping.csv')
blocks = {0: 'pred', 1: 'rep'}
num_blocks = len(blocks)
num_subjects = data_mapping['subject'].unique().shape[0]
base_data_folder = os.path.expanduser('~/Audio2AFC_CP/processed/')
plot_data_folder = base_data_folder + 'plots/'


def make_processed_filename(subj_number, block_str, prefix='', suffix='processed'):
    """
    builds the full path to the file that contains the valid trials for a given subject and block
    :param subj_number: int
    :param block_str: str, either 'rep' or 'pred'
    :param prefix: str, what should be prepended to filename
    :param suffix: str, what should be appended to filename, right before .csv extension
    :return: str
    """
    assert block_str in blocks.values()

    d = pd.read_csv('../data_mapping.csv')
    timestamp_as_df = d.loc[(d['subject'] == subj_number) & (d['block'] == block_str)]
    timestamp = timestamp_as_df.iloc[0]['timestamp']  # a string

    return base_data_folder + timestamp + '/' + prefix + str(subj_number) + block_str + suffix + '.csv'


def read_valid_trials(subj_number, block_str):
    """

    :param subj_number: int (count starts at 1, not 0!)
    :param block_str: str, should be in blocks.values()
    :return: returns pandas.DataFrame
    """
    assert block_str in blocks.values()
    return pd.read_csv(make_processed_filename(subj_number, block_str, prefix='pilot', suffix='_FIRA_valid_trials'))


if __name__ == '__main__':
    pass
