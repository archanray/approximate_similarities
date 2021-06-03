% This script generates text embedding for a p.d. text kernel constructed
% from data-dependent random features map using alignment-aware distance 
% for measuring the similairity between two sentences/documents. 
%
% Author: Lingfei Wu
% Date: 11/28/2018

clear,clc
parpool('local');

addpath(genpath('utilities'));
file_dir = './WME_tc_datasets_emnlp18';
filename = 'twitter';
filename_postfix = '-emd_tr_te_split.mat';
disp(filename);

gamma = 1; % if gamma=-1, use wmd distance directly; 
          % otherwise, use exp(-gamma*wmd)
R = 128; % number of random documents
DMin = 1; % minimum number of random words in a random document
DMax = 6; % maximum number of random words in a random document
dataSplit = 1; % we have total 5 different data splits for Train/Test
randdoc_scheme = 2; % if 1, leverage score
wordweight_scheme = 1; % if 1, use nbow

% load data and generate corresponding train and test data
Data = load(strcat(file_dir,'/',filename,filename_postfix));
[val_min,val_max,d,nbow_X_allDoc,idf_X_allDoc,tf_idf_X_allDoc] = ...
    wme_GenFea_preproc(Data);

[trainData, testData, telapsed_fea_gen] = wme_GenFea_leverage(Data,...
        gamma,R,dataSplit,...
        nbow_X_allDoc,idf_X_allDoc,tf_idf_X_allDoc,...
        randdoc_scheme,wordweight_scheme);
% csvwrite(strcat(file_dir,'/',filename,'_wme_Train'), trainData);
% csvwrite(strcat(file_dir,'/',filename,'_wme_Test'), testData);

telapsed_fea_gen

delete(gcp);

%     real_total_emd_time: 5.5285
%           user_emd_time: 6.8411
%     user_train_emd_time: 2.1764
%      user_test_emd_time: 4.6647