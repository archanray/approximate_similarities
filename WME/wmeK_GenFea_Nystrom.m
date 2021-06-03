% This script generates text embedding for a p.d. text kernel constructed
% from data-dependent random features map using alignment-aware distance 
% for measuring the similairity between two sentences/documents.
% Here, we need to compute ground distance for every pair of unique words 
% in order to compute WMD. This is not efficient since there are a lot of 
% redundent computations. 
%
% Author: Lingfei Wu
% Date: 11/28/2018

function [Train,Test,Runtime] = wmeK_GenFea_Nystrom(Data,...
    gamma,R,dataSplit,...
    nbow_X_allDoc,idf_X_allDoc,tf_idf_X_allDoc,...
    randdoc_scheme,wordweight_scheme, sample_size, mode, c_for_eig)

    if size(Data.TR,1) == 1
        dataSplit = 1;
    end
    train_words = Data.words(Data.TR(dataSplit,:));
    train_BOW_X = Data.BOW_X(Data.TR(dataSplit,:));
    train_X = Data.X(Data.TR(dataSplit,:));
    train_Y = Data.Y(Data.TR(dataSplit,:));
    test_words = Data.words(Data.TE(dataSplit,:));
    test_BOW_X = Data.BOW_X(Data.TE(dataSplit,:));
    test_X = Data.X(Data.TE(dataSplit,:));
    test_Y = Data.Y(Data.TE(dataSplit,:));

    % get nbow and tf-idf weights
    train_NBOW_X = nbow_X_allDoc(Data.TR(dataSplit,:));
    train_IDF_X = idf_X_allDoc(Data.TR(dataSplit,:));
    train_TFIDF_X = tf_idf_X_allDoc(Data.TR(dataSplit,:));
    test_NBOW_X = nbow_X_allDoc(Data.TE(dataSplit,:));
    test_TFIDF_X = tf_idf_X_allDoc(Data.TE(dataSplit,:));

    timer_start = tic;
    if wordweight_scheme == 1 % use NBOW
        train_weight_X = train_NBOW_X;
        test_weight_X = test_NBOW_X;
    elseif wordweight_scheme == 2 % use TFIDF
        train_weight_X = train_TFIDF_X;
        test_weight_X = test_TFIDF_X;
    end
    if strcmp(mode, 'nystrom')
        if randdoc_scheme == 1 && sample_size > -1
            % use uniform distribution
            samples = [];
            sample_weights = [];
            [trainFeaX_random, samples, sample_weights, train_emd_time] = WMD_Nystrom(train_X, train_weight_X, gamma, sample_size, R, samples, sample_weights, c_for_eig);
            fprintf('Finish computing trainFeaX \n');
            [testFeaX_random, samples, sample_weights, test_emd_time] = WMD_Nystrom(test_X, test_weight_X, gamma, sample_size, R, samples, sample_weights, c_for_eig);
            fprintf('Finish computing testFeaX \n');
        end
    end
    if strcmp(mode, 'CUR')
        if randdoc_scheme == 1 && sample_size > -1
            % use uniform distribution
            samples = [];
            sample_weights = [];
            [trainFeaX_random, samples, sample_weights, train_emd_time] = WMD_CUR(train_X, train_weight_X, gamma, sample_size, samples, sample_weights);
            fprintf('Finish computing trainFeaX \n');
            [testFeaX_random, samples, sample_weights, test_emd_time] = WMD_CUR(test_X, test_weight_X, gamma, sample_size, samples, sample_weights);
            fprintf('Finish computing testFeaX \n');
        end
    end
    
    if strcmp(mode, 'CUR_alt')
        if randdoc_scheme == 1 && sample_size > -1
            % use uniform distribution
            samples = [];
            sample_weights = [];
            [trainFeaX_random, samples1, sample_weights1, samples2, sample_weights2, train_emd_time] = ...
                WMD_CUR_2(train_X, train_weight_X, gamma, sample_size, samples, sample_weights, [], []);
            fprintf('Finish computing trainFeaX \n');
            [testFeaX_random, samples1, sample_weights1, samples2, sample_weights2, test_emd_time] = ...
                WMD_CUR_2(test_X, test_weight_X, gamma, sample_size, samples1, sample_weights1, samples2, sample_weights2);
            fprintf('Finish computing testFeaX \n');
        end
    end
    
    if strcmp(mode,'optimal')
        [trainFeaX_random, train_emd_time] = wmd_dist(train_X,train_weight_X,...
                                                train_X,train_weight_X,gamma);
        fprintf('Finish computing trainFeaX \n');
        [testFeaX_random, test_emd_time] = wmd_dist(test_X,test_weight_X,...
                                                test_X,test_weight_X,gamma);
        fprintf('Finish computing testFeaX \n');
        % optimal low rank for a given size
        trainFeaX_random = optimal_rank(trainFeaX_random, sample_size);
        fprintf('Finish computing optimal trainFeaX \n');
        testFeaX_random = optimal_rank(testFeaX_random, sample_size);
        fprintf('Finish computing optimal testFeaX \n');
    end

    if randdoc_scheme == 1 && sample_size == -1
        % original code and processing
        [trainFeaX_random, train_emd_time] = wmd_dist(train_X,train_weight_X,...
                                                train_X,train_weight_X,gamma);
        fprintf('Finish computing trainFeaX \n');
        [testFeaX_random, test_emd_time] = wmd_dist(test_X,test_weight_X,...
                                                test_X,test_weight_X,gamma);
        fprintf('Finish computing testFeaX \n');
        trainFeaX_random = trainFeaX_random/sqrt(R); 
        testFeaX_random = testFeaX_random/sqrt(R);
    
        % house cleaning
        trainFeaX_random(trainFeaX_random > 1e+08) = 0;
    end
    
    Train = [train_Y', trainFeaX_random];
    Test = [test_Y', testFeaX_random];
    telapsed_random_fea_gen = toc(timer_start);
    
    % Note: real_total_end_time is the real total time, including both emd
    % and ground distance, of generating both train and test features using 
    % multithreads. user_emd_time is the real time that accounts for 
    % computation of emd with one thread. 
    Runtime.real_total_emd_time = telapsed_random_fea_gen;
    Runtime.user_emd_time = train_emd_time + test_emd_time;
    Runtime.user_train_emd_time = train_emd_time;
    Runtime.user_test_emd_time = test_emd_time;
end
