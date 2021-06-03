% This script generates text embedding for a p.d. text kernel constructed
% from data-independent random features map using alignment-aware distance 
% for measuring the similairity between two sentences/documents. 
% We use Liblinear to perform grid search with K-fold cross-validation!
%
% Author: Lingfei Wu
% Date: 11/28/2018

clear,clc
parpool('local', 'IdleTimeout', Inf);

addpath(genpath('utilities'));
addpath('liblinear-2.41\matlab');

% file_dir = './data_proc';
file_dir = './WME_tc_datasets_emnlp18' ;
filename_list = {'twitter'};

randdoc_scheme = 1;     % if 1, RF features - uniform distribution
wordemb_scheme = 2;     % if 1, use pre-trained word2vec
                        % if 2, use pre-trained gloVe
                        % if 3, use pre-trained psl
wordweight_scheme = 1;  % if 1, use nbow
docemb_scheme = 2;      % if 1, use dist directly; 
                        % if 2, use soft-min of dist
                       
if docemb_scheme == 2
    gamma_list = [1e-3 1e-2 5e-2 0.10 0.5 1.0 1.5];
elseif docemb_scheme == 1
    gamma_list = -1;
end

sample_size_list = [100:20:1500];
%
%R = 256; % number of random documents generated
dataSplit = 1; % we have total 5 different data splits for Train/Test
CV = 10; % number of folders of cross validation
% this list will save the mean averages at any loop
parax = []; 
parax_std = [];
telapsed_something = [];
telapsed_linear_network = [];
counter_parax = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for jjj = 1:length(filename_list)
    filename = filename_list{jjj};
    disp(filename);
    if strcmp(filename, 'twitter')
        filename_postfix = '-emd_tr_te_split.mat';
    end   
    
    % load the train data
    timer_start = tic;
    Data = load(strcat(file_dir,'/',filename,filename_postfix));
    TR_index = Data.TR;
    if size(TR_index,1) == 1
        dataSplit = 1;
    end
    train_words = Data.words(TR_index(dataSplit,:));
    train_BOW_X = Data.BOW_X(TR_index(dataSplit,:));
    train_X = Data.X(TR_index(dataSplit,:));
    train_Y = Data.Y(TR_index(dataSplit,:));
    telapsed_data_load = toc(timer_start)

    [val_min,val_max,d,nbow_X_allDoc,idf_X_allDoc,tf_idf_X_allDoc] = ...
        wme_GenFea_preproc(Data);
    train_NBOW_X = nbow_X_allDoc(Data.TR(dataSplit,:));
    train_IDF_X = idf_X_allDoc(Data.TR(dataSplit,:));
    train_TFIDF_X = tf_idf_X_allDoc(Data.TR(dataSplit,:));
    
    info.aveAccu_best = 0;
    info.valAccuHist = [];
    info.sample_size_Hist = [];
    info.lambda_invHist = [];
    for jj = 1:length(sample_size_list)
%     for c_for_eig = 0.2:0.2:1.8
    for c_for_eig = 1:1
    for j = 1:length(gamma_list)
        %==================================================================
%         R = sample_size_list(jj);
        R = 1;
        sample_size = sample_size_list(jj)
        %==================================================================
        % DMax = DMax_list(jj)
        gamma = gamma_list(j)
       
        % shuffle the train data
        shuffle_index = randperm(length(train_Y)); 
        X = train_X(shuffle_index);
        Y = train_Y(shuffle_index);
        NBOW_X = train_NBOW_X(shuffle_index);
        IDF_X = train_IDF_X(shuffle_index);
        TFIDF_X = train_TFIDF_X(shuffle_index);
        N = size(X,2);
        %==================================================================
        trainData = zeros(N,R+1);
        %==================================================================
        rng('default')
        timer_start = tic;
        
        if wordweight_scheme == 1 % use NBOW
            weight_X = NBOW_X;
        elseif wordweight_scheme == 2 % use TFIDF
            weight_X = TFIDF_X;
        end
        
%         trainFeaX_random = wmd_dist(X,weight_X,X,weight_X,gamma);
%         
%         [trainFeaX_random, error_train] = Nystrom(trainFeaX_random, sample_size);
%         trainFeaX_random = abs(trainFeaX_random);
%         
%         trainFeaX_random = trainFeaX_random/sqrt(R); 
        samples = [];
        sample_weights = [];
        % uncomment for wmd nystrom
%         [trainFeaX_random, samples, sample_weights, train_emd_time] = WMD_Nystrom(X, weight_X, ...
%                                       gamma, sample_size, R, samples, sample_weights, c_for_eig);
        % uncomment for wmd CUR
        [trainFeaX_random, samples, sample_weights, train_emd_time] = WMD_CUR(X, weight_X, ...
                                      gamma, sample_size, samples, sample_weights);
                                  
%         [trainFeaX_random, samples1, sample_weights1, samples2, sample_weights2, train_emd_time] = WMD_CUR_2(X, weight_X, ...
%                                       gamma, sample_size, [], [], [], []);
        % uncomment for wmd optimal
%         [trainFeaX_random, train_emd_time] = wmd_dist(X,weight_X,...
%                                                 X,weight_X,gamma);
%         trainFeaX_random = optimal_rank(trainFeaX_random, sample_size);
        fprintf('Finish computing trainFeaX \n');
%         [testFeaX_random, test_emd_time] = WMD_Nystrom(test_X, test_weight_X, ...
%                                         gamma, sample_size, R);
%         fprintf('Finish computing testFeaX \n');
        
        trainData(:,2:end) = trainFeaX_random;
        trainData(:,1) = Y;
        telapsed_fea_gen = toc(timer_start);

        disp('------------------------------------------------------');
        disp('LIBLinear performs basic grid search by varying lambda');
        disp('------------------------------------------------------');
        % Linear Kernel
        %lambda_inverse = [1e2 3e2 5e2 8e2 1e3 3e3 5e3 8e3 1e4 3e4 5e4 8e4...
        %     1e5 3e5 5e5 8e5 1e6 1e7 1e8 1e9 1e10 1e11];
        lambda_inverse = [10000:5000:100000];
        for i=1:length(lambda_inverse)
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
            counter_parax = counter_parax+1;
            parax(counter_parax) = ave_valAccu;
            parax_std(counter_parax) = std_valAccu;
            telapsed_something(counter_parax) = telapsed_fea_gen;
            telapsed_linear_network(counter_parax) = telapsed_liblinear;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if(info.aveAccu_best+0.1 < ave_valAccu)
                % info.DMaxHist = [info.DMaxHist;DMax];
                info.sample_size_Hist = [info.sample_size_Hist;sample_size];
                info.lambda_invHist = [info.lambda_invHist;lambda_inverse(i)];
                info.valAccuHist = [info.valAccuHist;valAccu];
                info.valAccu = valAccu;
                info.aveAccu_best = ave_valAccu;
                info.stdAccu = std_valAccu;
                info.telapsed_fea_gen = telapsed_fea_gen;
                info.telapsed_liblinear = telapsed_liblinear;
                info.runtime = telapsed_fea_gen + telapsed_liblinear;
                info.gamma = gamma;
                info.R = R;
                %info.DMin = DMin;
                %info.DMax = DMax;
                info.sample_size = sample_size;
                info.lambda_inverse = lambda_inverse(i);
                info.randdoc_scheme = randdoc_scheme;
                info.wordemb_scheme = wordemb_scheme;
                info.wordweight_scheme = wordweight_scheme;
                info.docemb_scheme = docemb_scheme;
                info.c_for_eig = c_for_eig;
            end
        end
    end
    end
    end
    disp(info);
    savefilename = [filename '_Krd' num2str(randdoc_scheme) ...
        '_we' num2str(wordemb_scheme) '_ww' num2str(wordweight_scheme)...
        '_de' num2str(docemb_scheme) '_R' num2str(R) '_'  num2str(CV) 'fold_CV'];
    save(savefilename,'info');
end
delete(gcp);
