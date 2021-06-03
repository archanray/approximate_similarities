% This script generates text embedding for a p.d. text kernel constructed
% from data-dependent random features map using alignment-aware distance 
% for measuring the similairity between two sentences/documents. 
% Expts B: investigate performance changes when varying R using the 
% parameters learned from 10-folds cross validation with R = 256.
%
% Author: Lingfei Wu
% Date: 11/28/2018

clear,clc
nthreads = 10; % set as many as your cpu cores
parpool('local', nthreads);

addpath(genpath('utilities'));
addpath('liblinear-2.41\matlab');
file_dir = './WME_tc_datasets_emnlp18';
filename_list = {'ohsumed'};
apprx_mode = 'CUR';

randdoc_scheme = 1;     % if 1, RF features - uniform distribution
wordemb_scheme = 2;     % if 1, use pre-trained word2vec
                        % if 2, use pre-trained gloVe
                        % if 3, use pre-trained psl
wordweight_scheme = 1;  % if 1, use nbow
docemb_scheme = 2;      % if 1, use dist directly; 
                        % if 2, use soft-min of dist
                       
DMin = 1;    
% R_list = [4 8 16 32 64 128 256 512];
% R_list = [4 8 16 32 64 128 256 512 1024 2048 4096 8192];
sample_size_list = [2500];
R = 1;
c = 1;
for jjj = 1:length(filename_list)
    dataSplit_list = [1 2 3 4 5]; % total 5 different splits for Train/Test
    info = [];
    filename = filename_list{jjj};
    disp(filename);
    if strcmp(filename, filename_list{1})
        if wordemb_scheme >= 1 
            if wordweight_scheme == 1
                if docemb_scheme == 1
                    % DMax = 9;
                    gamma = -1;
                    lambda_inverse = 1000;
                elseif docemb_scheme == 2
                    % DMax = 9;
                    gamma = 1.2;
                    lambda_inverse = 23127765;
                end  
            end
        end
        filename_postfix = '-emd_tr_te_ix.mat';
    end
    disp(filename_postfix);
    
    % load data and generate corresponding train and test data
    Data = load(strcat(file_dir,'/',filename,filename_postfix));
    [val_min,val_max,d,nbow_X_allDoc,idf_X_allDoc,tf_idf_X_allDoc] = ...
        wme_GenFea_preproc(Data);
    
    Accu_best_list = zeros(2*length(dataSplit_list),length(sample_size_list));
    telapsed_liblinear_list = zeros(1*length(dataSplit_list),length(sample_size_list));
    real_total_emd_time_list = zeros(1*length(dataSplit_list),length(sample_size_list));
    real_user_emd_time_list = zeros(1*length(dataSplit_list),length(sample_size_list));
    for jj = 1:length(dataSplit_list)
        dataSplit = dataSplit_list(jj);
        for j = 1:length(sample_size_list)
            sample_size = sample_size_list(j)
            gamma
            [trainData, testData, telapsed_fea_gen] = wmeK_GenFea_Nystrom(Data,...
                gamma,R,dataSplit,...
                nbow_X_allDoc,idf_X_allDoc,tf_idf_X_allDoc,...
                randdoc_scheme,wordweight_scheme,sample_size, apprx_mode, c);
            

            disp('------------------------------------------------------');
            disp('LIBLinear on WME by varying number of random documents');
            disp('------------------------------------------------------');
            trainFeaX = trainData(:,2:end);
            trainy = trainData(:,1);
            testFeaX = testData(:,2:end);
            testy = testData(:,1);

            % Linear Kernel
            timer_start = tic;
            s2 = num2str(lambda_inverse);
%             s1 = '-s 2 -e 0.0001 -q -c '; % liblinear
            s1 = '-s 2 -e 0.09366 -n 8 -q -c '; % liblinear with omp
            s = [s1 s2];
            model_linear = train(trainy, sparse(trainFeaX), s);
            [train_predict_label, train_accuracy, train_dec_values] = ...
                    predict(trainy, sparse(trainFeaX), model_linear);
            [test_predict_label, test_accuracy, test_dec_values] = ...
                predict(testy, sparse(testFeaX), model_linear);
            telapsed_liblinear = toc(timer_start);
            Accu_best_list(2*(jj-1)+1,j) = train_accuracy(1);
            Accu_best_list(2*(jj-1)+2,j) = test_accuracy(1);
            telapsed_liblinear_list(jj,j) = telapsed_liblinear;
            real_total_emd_time_list(jj,j) = telapsed_fea_gen.real_total_emd_time;
            real_user_emd_time_list(jj,j) = telapsed_fea_gen.user_emd_time/nthreads;
        end
    end
    info.Accu_best_train_ave = mean(Accu_best_list(1:2:end,:),1);
    info.Accu_best_train_std = std(Accu_best_list(1:2:end,:),1);
    info.Accu_best_test_ave = mean(Accu_best_list(2:2:end,:),1);
    info.Accu_best_test_std = std(Accu_best_list(2:2:end,:),1);
    info.Accu_best_list = Accu_best_list;
    info.real_total_emd_time_list = real_total_emd_time_list;
    info.real_user_emd_time_list = real_user_emd_time_list;
    info.telapsed_liblinear = telapsed_liblinear_list;
    % info.R = R_list;
    % info.DMin = DMin;
    % info.DMax = DMax;
    info.sample_size = sample_size_list;
    info.gamma = gamma;
    info.lambda_inverse = lambda_inverse;
    info.randdoc_scheme = randdoc_scheme;
    info.wordemb_scheme = wordemb_scheme;
    info.wordweight_scheme = wordweight_scheme;
    info.docemb_scheme = docemb_scheme;
    disp(info);
    savefilename = [filename '_Krd' num2str(randdoc_scheme) ...
        '_we' num2str(wordemb_scheme) '_ww' num2str(wordweight_scheme)...
        '_de' num2str(docemb_scheme) '_VaryingR_allSplits_CV_R256'];
    save(savefilename,'info')

end
delete(gcp);
