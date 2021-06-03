% generate all traindata
Data = load(strcat('WME_tc_datasets_emnlp18','/','twitter','-emd_tr_te_split.mat'));
TR_index = Data.TR;


timer_start = tic;
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

gamma = 1;
X = train_X(shuffle_index);
