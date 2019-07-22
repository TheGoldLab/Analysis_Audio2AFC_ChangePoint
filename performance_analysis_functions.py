import pandas as pd
import matplotlib.pyplot as plt
import os, sys
import numpy as np

# def produce_df(file_name):
#     """

#     """
#     data = pd.read_csv(file_name)
    

 


#     # Adding another column to indicate the number of trials that has passed since the last source switch occurred.
#     positionCol = [1] * data.count()
#     data['posSinceCP'] = positionCol
    
#     # Add the hazard rate column. 
#     # Load the sim_data.csv file.
#     simulation = os.path.expanduser(
#             '~/Documents/GitHub/Analysis_Audio2AFC_ChangePoint/sim_data.csv'
#     )
    
#     simulation_data = pd.read_csv(simulation)
#     data = data.merge(simulation_data['hazard'], left_index=True, right_index=True)
    
#     # Iterate over the rows in the source, checking the previous row and comparing it to the first one to see if there is 
#     # a source switch.
#     # This fills out the sourceSwitch column.

#     data['sourceChange'] = data['source'].shift(1)
#     data['sourceSwitch'] = data['source'] != data['sourceChange']
    
   
#     # Iterate over the rows in the sourceSwitch column and fill the posSinceCP column. 
#     #If there has been a switch, set the counter to 0, else, keep on iterating the counter. 
    
#     for row_number in range(len(data)):

#         isSwitch = data.iloc[row_number, 'sourceSwitch']
#         if isSwitch:
#             count_switches = 0
#         else:
#             count_switches+=1
#         data.iloc[row_number, 'posSinceCP'] = count_switches
        
#     return data


# def extract_hazard(data):
#     """
    
#     """
#     # Finding the maximum value in the posSinceCP column to find the range of iterations.

#     high_hazard_extracted = data[data['hazard'] == 0.9].copy()
#     low_hazard_extracted = data[data['hazard'] == 0.1].copy()
    
    


        
#     max_value_high = high_hazard_extracted['posSinceCP'].max()
#     max_value_low = low_hazard_extracted['posSinceCP'].max()
    
    
#     #List to keep track of the percentage correct per value of the position counter.
#     list_of_percentages_low = []
#     list_of_percentages_high = []

#     # Calculates % correct per each value of position counter: number of correct answers / number of trials (rows).
#     # For high hazard rate.
#     for val in range(0, int(max_value_high+1)):

#         extracted_df_high = data[(data['posSinceCP'] == val) & (data['hazard']==0.9)].copy()
#         percent_correct_high = extracted_df_high['correct'].sum() / len(extracted_df_high)
#         list_of_percentages_high.append(percent_correct_high)
        

#     # Calculates % correct per each value of position counter: number of correct answers / number of trials (rows).
#     # For low hazard rate.
#     for val in range(0, int(max_value_low+1)):

#         extracted_df_low = data[(data['posSinceCP'] == val) & (data['hazard']==0.1)].copy()
#         percent_correct_low = extracted_df_low['correct'].sum() / len(extracted_df_low)
#         list_of_percentages_low.append(percent_correct_low)
        
#     vector_high = np.arange(0,(max_value_high+1))
#     vector_low = np.arange(0, (max_value_low+1))
    
        
#     return list_of_percentages_high, list_of_percentages_low, vector_high, vector_low


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

# def add_switch_col(df):
    
#     # Adding a column to indicate source switches.
# #     df['sourceSwitch'] = np.nan
# #     df['sourceShift'] = df['source'].shift(1)
# #     df['sourceSwitch'] = df['source'] != df['sourceShift']
# #     df = df.drop(['sourceShift'], axis=1)

#     return df

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
    # For high hazard rate.
    for val in range(0, int(max_value+1)):

        extracted_df = df[df['posSinceCP'] == val].copy()
        percent_correct = extracted_df['correct'].sum() / len(extracted_df)
        list_of_percentages.append(percent_correct)
        
        
    vector = np.arange(0,(max_value+1))
    assert len(vector) == len(list_of_percentages)

    return list_of_percentages, vector

