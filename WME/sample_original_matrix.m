function [trainData] = sample_original_matrix(trainData, max_samples)
    k = ceil(size(trainData,1)/max_samples);
    cv = cvpartition(trainData(:,1),'KFold',k);
    dataMat = trainData(:,2:end);
    dataMat = dataMat(cv.test(1), cv.test(1));
    labels = trainData(cv.test(1),1);
    trainData = [labels, dataMat];
end