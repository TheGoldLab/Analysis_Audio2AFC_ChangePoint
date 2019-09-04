"""
For all subjects and blocks, this script provides info about the
trialIndex of valid trials as a table with following columns:
subject, block, minIdx, maxIdx, repeated, skipped
"""
import pandas as pd
import data_processing as dp

def make_html_table():
    """
    1. get number of subjects by reading data_mapping.csv
    2. loop through subjects and blocks and
       a) get trial indices
    :return:
    """

    for subj in range(dp.num_subjects):
        subject_number = subj + 1
        for b in range(dp.num_blocks):
            block = dp.blocks[b]
            data = dp.get_valid_trials(subj + 1, block)


    row = {'subject': 1,
           'block': 'pred',
           'minIdx': 2,
           'maxIdx': 387,
           'repeated': [3, 3, 5, 5],
           'skipped': [7]}


if __name__ == '__main__':
    make_html_table()