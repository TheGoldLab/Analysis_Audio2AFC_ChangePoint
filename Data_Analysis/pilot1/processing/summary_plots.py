import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import data_processing as dp

font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 18}

mpl.rc('font', **font)
figsz = (6*2.5, 8)  # width and height in inches

output_filename = 'summary_plots.png'

pcorrect = {}
num_valid_trials = {}


def make_plot():
    """
    If N subjects have each done a report and a prediction block,
    we make an N by 2 grid of plots, with trial sequential number on x axis, trialStart timestamp on y axis.
    An annotation states in the upper-left corner how many valid trials out of the total number of trials.
    Also annotate percent correct on block.
    """
    fig, ax = plt.subplots(dp.num_subjects, dp.num_blocks, figsize=figsz, sharey=True, sharex=True)
    for subj in range(dp.num_subjects):
        for block in range(dp.num_blocks):
            block_str = dp.blocks[block]
            data = dp.get_valid_trials(subj+1, block_str)
            gaps = np.diff(data['trialStart'])
            num_valid_trials[(subj, block)] = data.shape[0]
            pcorrect[(subj, block)] = (data['correct'].sum() / num_valid_trials[(subj, block)]) * 100
            curr_ax = ax[subj, block]
            curr_ax.plot(range(len(gaps)), gaps)
            if subj == dp.num_subjects - 1:
                curr_ax.set_xlabel('valid trial index')
            if block == 0:
                curr_ax.set_ylabel(f'subject {subj + 1} \ntime gap (s)')
            if subj == 0:
                curr_ax.set_title(dp.blocks[block])

    # separate loop to set annotations
    for subj in range(dp.num_subjects):
        for block in range(dp.num_blocks):
            curr_ax = ax[subj, block]
            xrng = curr_ax.get_xlim()
            yrng = curr_ax.get_ylim()
            annotation = 'valid {:d}, correct {:.1f}%'.format(num_valid_trials[(subj, block)], pcorrect[(subj, block)])
            annotation_xpos = xrng[0] + 0.1 * np.diff(xrng)
            annotation_ypos = yrng[0] + 0.9 * np.diff(yrng)
            curr_ax.text(annotation_xpos, annotation_ypos, annotation)

    plt.savefig(dp.plot_data_folder + output_filename, bbox_inches='tight')


if __name__ == '__main__':
    make_plot()
