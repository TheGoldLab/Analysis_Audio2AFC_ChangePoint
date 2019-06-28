"""
If N subjects have each done a report and a prediction block,
we make an N by 2 grid of plots, with trial sequential number on x axis, trialStart timestamp on y axis.
An annotation states in the upper-left corner how many valid trials out of the total number of trials.
Also annotate percent correct on block.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 18}

mpl.rc('font', **font)
figsz = (6*2.5, 8)  # width and height in inches


output_filename = 'summary_plots.png'
num_subjects = 2
num_blocks = 2
base_data_folder = os.path.expanduser('~/Audio2AFC_CP/processed/')
block_strings = {0: 'rep', 1: 'pred'}

pcorrect = {}
num_valid_trials = {}


def make_filename(subj_number, block_str):
    """
    builds the full path to the file that contains the valid trials for a given subject and block
    :param subj_number: int
    :param block_str: str, either 'rep' or 'pred'
    :return: str
    """
    d = pd.read_csv('../data_mapping.csv')
    timestamp_as_df = d.loc[(d['subject'] == subj_number) & (d['block'] == block_str)]
    timestamp = timestamp_as_df.iloc[0]['timestamp']  # a string
    return base_data_folder + timestamp + '/pilot' + str(subj_number) + block_str + '_FIRA_valid_trials.csv'


def make_plot():
    fig, ax = plt.subplots(num_subjects, num_blocks, figsize=figsz, sharey=True, sharex=True)
    for subj in range(num_subjects):
        for block in range(num_blocks):
            block_str = block_strings[block]
            data = pd.read_csv(make_filename(subj + 1, block_str))
            gaps = np.diff(data['trialStart'])
            num_valid_trials[(subj, block)] = data.shape[0]
            pcorrect[(subj, block)] = (data['correct'].sum() / num_valid_trials[(subj, block)]) * 100
            curr_ax = ax[subj, block]
            curr_ax.plot(range(len(gaps)), gaps)
            if subj == num_subjects - 1:
                curr_ax.set_xlabel('valid trial index')
            if block == 0:
                curr_ax.set_ylabel(f'subject {subj + 1} \ntime gap (s)')
            if subj == 0:
                curr_ax.set_title(block_strings[block])

    # separate loop to set annotations
    for subj in range(num_subjects):
        for block in range(num_blocks):
            curr_ax = ax[subj, block]
            xrng = curr_ax.get_xlim()
            yrng = curr_ax.get_ylim()
            annotation = 'valid {:d}, correct {:.1f}%'.format(num_valid_trials[(subj, block)], pcorrect[(subj, block)])
            annotation_xpos = xrng[0] + 0.1 * np.diff(xrng)
            annotation_ypos = yrng[0] + 0.9 * np.diff(yrng)
            curr_ax.text(annotation_xpos, annotation_ypos, annotation)

    plt.savefig(base_data_folder + 'plots/' + output_filename, bbox_inches='tight')


if __name__ == '__main__':
    make_plot()
