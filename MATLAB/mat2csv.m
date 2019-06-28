function mat2csv(subj_number, task_str)
% this function is the standard pre-processing step to convert the data that
% we need for our analysis from .mat to .csv format.
% ARGS:
% subj_number        should be a string
% task_str           a string, either 'rep' or 'pred'
% NOTES: 
%    first col of data_mapping.csv expected to be timestamp
%    tbUseProject('Analysis_Audio2AFC_ChangePoint') should be called before

%% Folders and path variables

studyTag = 'Audio2AFC_CP';

% mapping from (subj_number, task_str) to timestamp is contained in
% data_mapping.csv
mapping_data = readtable('data_mapping.csv', 'Delimiter', ',', ...
    'ReadVariableNames', true);
data_timestamp = mapping_data{mapping_data.subject == subj_number & ...
    strcmp(mapping_data.block, task_str), 1};

% location of .csv files to output
csvPath = ['~/Audio2AFC_CP/raw/', data_timestamp{1},'/'];
fileNameWithoutExt = ['pilot', num2str(subj_number), task_str];

%% FIRA.ecodes data
[topNode, FIRA] = ...
    topsTreeNodeTopNode.loadRawData(studyTag,...
    data_timestamp{1});

for i=1:length(FIRA.ecodes.name)
    if strcmp(FIRA.ecodes.name{i}, 'catch')
        FIRA.ecodes.name{i} = 'isCatch';
    end
end

T=array2table(FIRA.ecodes.data, 'VariableNames', FIRA.ecodes.name);
writetable(T,[csvPath,fileNameWithoutExt,'_FIRA.csv'],'WriteRowNames',true)
end
