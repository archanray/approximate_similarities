% This script generates text embedding for a p.d. text kernel constructed
% from data-dependent random features map using alignment-aware distance 
% for measuring the similairity between two sentences/documents.
% Here, we need to compute ground distance for every pair of unique words 
% in order to compute WMD. This is not efficient since there are a lot of 
% redundent computations. 
%
% Author: Lingfei Wu
% Date: 11/28/2018

function [Train,Test,Runtime] = wme_GenFea_leverage(Data,...
    gamma,R,dataSplit,...
    nbow_X_allDoc,idf_X_allDoc,tf_idf_X_allDoc,...
    randdoc_scheme,wordweight_scheme)

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

    if wordweight_scheme == 1 % use NBOW
        train_weight_X = train_NBOW_X;
        test_weight_X = test_NBOW_X;
    elseif wordweight_scheme == 2 % use TFIDF
        train_weight_X = train_TFIDF_X;
        test_weight_X = test_TFIDF_X;
    end
    
    % generate random features based on emd distance between original texts
    % and random texts where random words are sampled in R^d word space
    c=0.2;
    disp("datashape");
    disp(size(train_X));
    timer_start = tic;
    rng('default')
    if randdoc_scheme == 1 
        % Method 1: use leverage score sampling using a rbf kernel made
        % with the same gamma as WMD, then approximate as whatever
        timer_start = tic;
        sample_X = cell(1,R);
        % create kernel function using the same gamma as WMD gamma
        % kFunc = @(X,rowInd,colInd) gaussianKernel(X,rowInd,colInd,gamma);
        kFunc = @(X, weight_X, rowInd, colInd) wmd_distFunc(X, weight_X, rowInd, colInd, gamma);
        [trainFeaX_random, samples, train_emd_time] = modifiedRecursiveNystrom(train_X, train_weight_X, R, kFunc);
        parfor i = 1:length(samples)
            %disp(samples(i));
            sample_X{i} = train_X{samples(i)};
        end
    end
    if randdoc_scheme == 2
        % Method 1: use leverage score sampling using a rbf kernel made
        % with the same gamma as WMD, then approximate as whatever
        timer_start = tic;
        sample_X = cell(1,R);
        % create kernel function using the same gamma as WMD gamma
        % kFunc = @(X,rowInd,colInd) gaussianKernel(X,rowInd,colInd,gamma);
        kFunc = @(X, weight_X, rowInd, colInd) wmd_distFunc(X, weight_X, rowInd, colInd, gamma);
        [trainFeaX_random, samples, train_emd_time] = modifiedRecursiveNystrom2(train_X, train_weight_X, R, kFunc);
        parfor i = 1:length(samples)
            %disp(samples(i));
            sample_X{i} = train_X{samples(i)};
        end
    end
    % assign sample weights accordingly
    sample_weight_X = cell(1,R);
    parfor i = 1:length(samples)
        sample_weight_X{i} = train_weight_X{samples(i)};
    end
    
    %disp("chosen sample size");
    %disp(samples);
    %disp(size(samples));
    
    % [trainFeaX_random, train_emd_time] = wmd_dist(train_X,train_weight_X,...
    %     sample_X,sample_weight_X,gamma);
    % fprintf('Finish computing trainFeaX \n');
    [testFeaX_random, test_emd_time] = wmd_dist(test_X,test_weight_X,...
        sample_X,sample_weight_X,gamma);
    fprintf('Finish computing testFeaX \n');
    trainFeaX_random = trainFeaX_random/sqrt(R); 
    testFeaX_random = testFeaX_random/sqrt(R);
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
