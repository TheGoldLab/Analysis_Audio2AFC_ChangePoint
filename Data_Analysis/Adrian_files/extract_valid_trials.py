"""
script to be run from terminal
  $ python extract_valid_trials <filename with absolute path>
"""
import sys
import pandas as pd
# import numpy as np


def extract_valid(filename):
    data = pd.read_csv(filename)

    # deal with columns
    unwanted_col_names = [
        'taskID',
        'randSeedBase',
        'fixationOn',
        'fixationBlue',
        'targetOn',
        'sourceOn',
        'feedbackOn'
    ]

    valid = data.copy()

    valid.drop(unwanted_col_names, axis=1, inplace=True)

    # deal with rows

    # delete rows with NaN choice values
    index_names = valid[valid['choice'].isna()].index
    valid.drop(index_names, inplace=True)

    # Get names of indexes for which column RT has negative value
    index_names = valid[valid['RT'] < 0].index
    # Delete these row indexes from data frame
    valid.drop(index_names, inplace=True)

    return valid


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('script takes filename as single command line argument')
    else:
        fname = sys.argv[1]
        valid_data = extract_valid(fname)

        ofname = fname[:-4] + '_valid_trials.csv'
        ofname = ofname.replace('raw', 'processed')
        print('writing file to ', ofname)
        valid_data.to_csv(ofname, index=False)
