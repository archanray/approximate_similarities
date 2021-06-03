% this code will do cross validation.
% it will take a slice of the original wmd exponentiated matrix
% and then subsample it first.
% we then do the cross validation on this matrix
% the idea is to finish cross validation within one day, so we can do
% experiemtns extensively
clc;clear;
% function [] = fast_code_for_nystrom(user_chosen_lambda)
lambda_end = 50000;
% load the full similarity matrix
data_dir = './mat_files/';
dataset = 'twitter_K_set1';
exec_ID = '1';

dataset_file = strcat(data_dir, dataset, '.mat');
load(dataset_file, 'trainData');
% trainData = S.trainData;
% in the above trainData is what we are after

% load path and set environment variables
addpath('liblinear-2.41\matlab');
% addpath('./MatlabProgressBar');
parpool('local', 'IdleTimeout', Inf);
max_samples_number = 1500;

% try
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % SAMPLE FROM ORIGINAL TRAINDATA
    max_samples = ceil(sqrt(max_samples_number*length(trainData)));
    
    length_of_original_matrix = length(trainData);
    available_samples = [2:length_of_original_matrix];
    chosen_samples = datasample(available_samples, max_samples, 'Replace',false);
    
    reduced_trainData = trainData(:, chosen_samples);
    reduced_trainData_labels = trainData(:,1);
    
    train_X = reduced_trainData;
    train_Y = reduced_trainData_labels;

    % HYPERPARAMETERS
    randdoc_scheme = 1;     % if 1, RF features - uniform distribution
    wordemb_scheme = 2;     % if 1, use pre-trained word2vec
                            % if 2, use pre-trained gloVe
                            % if 3, use pre-trained psl
    wordweight_scheme = 1;  % if 1, use nbow
    docemb_scheme = 2;      % if 1, use dist directly; 
                            % if 2, use soft-min of dist
    
    if docemb_scheme == 2
        gamma_list = [1e-2];%[1e-3 1e-2 5e-2 0.10 0.5 1.0 1.5];
    elseif docemb_scheme == 1
        gamma_list = -1;
    end
    sample_size_list = [100, 200];%[100:50:max_samples_number];
    c_for_eig = [0.5];%[0.2:0.4:1.8];
    lambda_inverse = [50000];%[10000:5000:lambda_end];
    
    avg_accuracy_per_round = zeros(length(sample_size_list), ...
        length(gamma_list)*length(c_for_eig)*length(lambda_inverse));
    avg_std_per_round = avg_accuracy_per_round;
    avg_telapsed_fea_gen = avg_accuracy_per_round;
    avg_telapsed_liblinear = avg_accuracy_per_round;
    
    CV = 2; % number of folders of cross validation
    % this list will save the mean averages at any loop
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % check progress
    %numIterations = length(sample_size_list);
    parfor counter_sample_size = 1:length(sample_size_list)
        %pause(0.1);
        %updateParallel([], pwd);
        
        counter_parax = 0;
        avg_accuracy_per_round_sub_array = zeros(1, length(gamma_list)*length(c_for_eig)*length(lambda_inverse));
        avg_std_per_round_sub_array = zeros(1, length(gamma_list)*length(c_for_eig)*length(lambda_inverse));
        avg_telapsed_fea_gen_sub_array = zeros(1, length(gamma_list)*length(c_for_eig)*length(lambda_inverse));
        avg_telapsed_liblinear_sub_array = zeros(1, length(gamma_list)*length(c_for_eig)*length(lambda_inverse));
        for counter_c = 1:length(c_for_eig)
            for counter_gamma = 1:length(gamma_list)
                %==================================================================
                % choose the sample size, gamma and c
                sample_size = sample_size_list(counter_sample_size);
                gamma = gamma_list(counter_gamma);
                c = c_for_eig(counter_c);
                %==================================================================
                % shuffle the train data
                shuffle_index = randperm(length(train_Y)); 
                X = train_X(shuffle_index, :);
                Y = train_Y(shuffle_index, :);
                
                membership_of_chosen_indices = ismember(shuffle_index, chosen_samples);
                rearranged_chosen_samples = find(membership_of_chosen_indices);
                
                N = size(X,2);
                %==================================================================
                rng('default')
                timer_start = tic;
                % get Nystrom features
                % [trainFeaX_random] = WMD_Nystrom_Once(X, gamma, sample_size, rearranged_chosen_samples, c);
                % get CUR features
                [trainFeaX_random] = WMD_CUR_Once(X, gamma, sample_size, rearranged_chosen_samples);
                %fprintf('Finish computing trainFeaX \n');
                telapsed_fea_gen = toc(timer_start);
                %==================================================================

                disp('------------------------------------------------------');
                disp('LIBLinear performs basic grid search by varying lambda');
                disp('------------------------------------------------------');
                % Linear Kernel
                %lambda_inverse = [1e2 3e2 5e2 8e2 1e3 3e3 5e3 8e3 1e4 3e4 5e4 8e4...
                %     1e5 3e5 5e5 8e5 1e6 1e7 1e8 1e9 1e10 1e11];
                for i=1:length(lambda_inverse)
                    counter_parax = counter_parax+1;
                    valAccu = zeros(1, CV);
                    for cv = 1:CV
                        subgroup_start = (cv-1) * floor(N/CV);
                        mod_remain = mod(N, CV);
                        div_remain = floor(N/CV);
                        if  mod_remain >= cv
                            subgroup_start = subgroup_start + cv;
                            subgroup_end = subgroup_start + div_remain;
                        else
                            subgroup_start = subgroup_start + mod_remain + 1;
                            subgroup_end = subgroup_start + div_remain -1;
                        end
                        test_indRange = subgroup_start:subgroup_end;  

                        train_indRange = setdiff(1:N,test_indRange);
                        trainFeaX = trainData(train_indRange,2:end);
                        trainy = trainData(train_indRange,1);
                        testFeaX = trainData(test_indRange,2:end);
                        testy = trainData(test_indRange,1);

                        s2 = num2str(lambda_inverse(i));
                        s1 = '-s 2 -e 0.0001 -q -c '; % liblinear
        %                 s1 = '-s 2 -e 0.0001 -n 8 -q -c '; % liblinear with omp
                        s = [s1 s2];
                        timer_start = tic;
                        model_linear = train(trainy, sparse(trainFeaX), s);
                        [test_predict_label, test_accuracy, test_dec_values] = ...
                            predict(testy, sparse(testFeaX), model_linear);
                        telapsed_liblinear = toc(timer_start);
                        valAccu(cv) = test_accuracy(1);             
                    end
                    ave_valAccu = mean(valAccu);
                    std_valAccu = std(valAccu);
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    avg_accuracy_per_round_sub_array(1, counter_parax) = ave_valAccu;
                    avg_std_per_round_sub_array(1, counter_parax) = std_valAccu;
                    avg_telapsed_fea_gen_sub_array(1, counter_parax) = telapsed_fea_gen;
                    avg_telapsed_liblinear_sub_array(1, counter_parax) = telapsed_liblinear;
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                end
            end
        end
        avg_accuracy_per_round(counter_sample_size,:) = avg_accuracy_per_round_sub_array;
        avg_std_per_round(counter_sample_size, :) = avg_std_per_round_sub_array;
        avg_telapsed_fea_gen(counter_sample_size, :) = avg_telapsed_fea_gen_sub_array;
        avg_telapsed_liblinear(counter_sample_size, :) = avg_telapsed_liblinear_sub_array
    end
    % progBar.release();
    savefilename = strcat('./validation_save_dir/', exec_ID, '_', string(lambda_end), '_', dataset);
    save(savefilename,  '-v7.3');
    % poolobj = gcp('nocreate');
%     delete(gcp);                        
% catch
    % poolobj = gcp('nocreate');
%     disp('execution failed');
    delete(gcp);
% end
% end