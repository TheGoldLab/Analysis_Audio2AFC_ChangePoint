% this script is the standard pre-processing step to convert the data that
% we need for our analysis from .mat to .csv format.
% 
clear all
tbUseProject('Analysis_Audio2AFC_ChangePoint');
subj_number = '2';               % should be a string
task_str = 'rep';                   % a string, either 'rep' or 'pred'
npilot = str2double(subj_number); % subject number as double
% clear classes
% clear mex
% clear

%% Folders and path variables

studyTag = 'Audio2AFC_CP'; 

% mapping of Pilot data to timestamps
% ======  DO NOT ERASE!! ======
% '2019_06_25_11_11' = subject 1 - prediction
% '2019_06_25_11_40' = subject 1 - report
% '2019_06_25_12_09' = subject 2 - prediction
% '2019_06_25_12_31' = subject 2 - report
% =============================
timestamp.subj1pred = '2019_06_25_11_11';
timestamp.subj1rep = '2019_06_25_11_40';
timestamp.subj2pred = '2019_06_25_12_09';
timestamp.subj2rep = '2019_06_25_12_31';

data_timestamp = timestamp.(['subj',subj_number,task_str]); 

% location of .csv files to output
csvPath = ['~/Audio2AFC_CP/raw/', data_timestamp,'/'];
fileNameWithoutExt = ['pilot', subj_number, task_str];

%% FIRA.ecodes data
[topNode, FIRA] = ...
    topsTreeNodeTopNode.loadRawData(studyTag,...
    data_timestamp);

for i=1:length(FIRA.ecodes.name)
    if strcmp(FIRA.ecodes.name{i}, 'catch')
        FIRA.ecodes.name{i} = 'isCatch';
    end
end

T=array2table(FIRA.ecodes.data, 'VariableNames', FIRA.ecodes.name);
writetable(T,[csvPath,fileNameWithoutExt,'_FIRA.csv'],'WriteRowNames',true)
