"""
script to generate trials
"""
import numpy as np
import os
import sys
# insert the path where mmcomplexity.py lives below
sys.path.append(os.path.expanduser('~/Git/GitHub/work/Analysis_Audio2AFC_ChangePoint/Python_modules'))
from mmcomplexity import *


np.random.seed(1)
h_values = [.1, .9]
num_blocks_per_h = 4
block_length = 205
block_name = 'Block00'

if __name__ == '__main__':
    block_number = 0
    for h in h_values:
        for b in range(num_blocks_per_h):
            block_number += 1
            sim = Audio2AFCSimulation(block_length, [h], 0, [1], catch_rate=0.05)
            sim.data.to_csv(block_name + str(block_number) + '.csv', index=False)
