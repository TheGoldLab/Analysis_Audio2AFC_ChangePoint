import pandas as pd
import matplotlib.pyplot as plt
import os, sys
import numpy as np



def get_timestamp(dataframe, subject, task_type):
    timestamp = dataframe[(dataframe['subject']==subject) & (dataframe['block']==task_type)]
    return timestamp.iloc[0,0]


def build_filename(task_type, subject):
    
    data_map = os.path.expanduser(
        '~/Documents/GitHub/Analysis_Audio2AFC_ChangePoint/data_mapping.csv'
    )

    read_data_map = pd.read_csv(data_map)
    
    if task_type == "rep":
         # Subject 1 Report Task 
        data = os.path.expanduser(
            '~/Documents/MATLAB/Audio2AFC_CP/processed/' +
            get_timestamp(read_data_map, subject+1, 'rep') + '/pilot' +
            str(subject+1) + 'rep_FIRA_valid_trials.csv'
        )
    elif task_type == "pred":
        # Subject 1 Prediction Task 
        data = os.path.expanduser(
            '~/Documents/MATLAB/Audio2AFC_CP/processed/' + get_timestamp(read_data_map, subject+1, 'pred') + '/pilot' +
            str(subject+1) + 'pred_FIRA_valid_trials.csv'
        )
     
    return data


def load_trial_seq():
    
    simulation = os.path.expanduser(
            '~/Documents/GitHub/Analysis_Audio2AFC_ChangePoint/sim_data.csv'
    )
    
    simulation_data = pd.read_csv(simulation)
    return simulation_data


def remove_cols(df):
    df = df.drop(["trialStart", "trialEnd", "RT", "direction", "isCatch", "unselectedTargetOff", "sound1On", "sound2On", 
                           "sound1Off", "sound2Off", "choiceTime", "secondChoiceTime", "targetOff", "fixationOff", "dirReleaseChoiceTime"], axis=1)
    return df


def compute_pos_data(data):
    
    data['posSinceCP'] = np.nan
    
    count_switches = 0

    for row_number in range(len(data)):

        isSwitch = data['source_switch'].iloc[row_number]
        if isSwitch:
            count_switches = 0
        else:
            count_switches+=1
        data.loc[data.index[row_number],'posSinceCP'] = count_switches
        
    return data
    
def merge_pos_sim(sim_data, df):
    df['mergeIndex'] = df['trialIndex'] - 1
    df = df.merge(sim_data[['hazard', 'source_switch', 'posSinceCP']], left_on="mergeIndex", right_index=True)
    return df
    
    
def add_hazard_and_switch_col(df):
    simulation_data = load_trial_seq()
    df['mergeIndex'] = df['trialIndex'] - 1
    df = df.merge(simulation_data[['hazard', 'source_switch']], left_on="mergeIndex", right_index=True)
    return df


def add_posSinceCP(df):
    df['posSinceCP'] = np.nan
    
    count_switches = 0

    for row_number in range(len(df)):

        isSwitch = df['source_switch'].iloc[row_number]
        if isSwitch:
            count_switches = 0
        else:
            count_switches+=1
        df.loc[df.index[row_number],'posSinceCP'] = count_switches
        
    return df



def extract_hazard(df, h):
    
    hazard_extracted = df[df['hazard'] == h].copy()
    
    return hazard_extracted

def compute_percent(df):
    
    max_value = df['posSinceCP'].max()
    
    #List to keep track of the percentage correct per value of the position counter.
    list_of_percentages = []


    # Calculates % correct per each value of position counter: number of correct answers / number of trials (rows).
    for val in range(0, int(max_value+1)):

        extracted_df = df[df['posSinceCP'] == val].copy()
        percent_correct = extracted_df['correct'].sum() / len(extracted_df)
        list_of_percentages.append(percent_correct)
        
        
    vector = np.arange(0,(max_value+1))
    assert len(vector) == len(list_of_percentages)

    return list_of_percentages, vector

