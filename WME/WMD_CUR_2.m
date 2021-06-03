% Code to compute nystrom on top of WMD features
% reuses the WMD library made available by  Lingfei Wu
%
% Author: Archan Ray
% Date: 19 October, 2020

function[mapped_feats, samples1, sample_weights1, samples2, sample_weights2, user_emd_time] = WMD_CUR_2(train_X,...
    train_weight_X, gamma, sample_size, samples1, sample_weights1, samples2, sample_weights2)
    
    % reduce set here before using Nystrom
    n = size(train_X,2);
    if size(samples1, 1) == 0
        % train case
        s1 = randsample(1:n, sample_size);
        s2 = randsample(1:n, ceil((sample_size)*0.5));
    end
    % support samples
    if size(samples1, 1) == 0 
        % train case
        samples2 = train_X(:, s2);
        sample_weights2 = train_weight_X(:, s2);
        samples1 = train_X(:, s1);
        sample_weights1 = train_weight_X(:, s1);
    end
    % compute the respective WMD values
    [Ks2, Ks2_emd_time] = wmd_dist(train_X,train_weight_X,...
        samples2,sample_weights2,gamma);
    [S1KS2, S1KS2_emd_time] = wmd_dist(samples1, sample_weights1,...
        samples2, sample_weights2,gamma);
    
    user_emd_time = Ks2_emd_time+S1KS2_emd_time;
    
    % consistency wih original code
    Ks2 = Ks2 / sqrt(1.0);
    S1KS2 = S1KS2 / sqrt(1.0);
    
    % house cleaning
    Ks2(Ks2 > 1e+08) = 0;
    S1KS2(S1KS2 > 1e+08) = 0;
    
    % compute the skeleton cur approximation feature
    if size(samples, 1) == 0
        % train case
        sKs = S1KS2;
    end
    [~,V,D] = svd(sKs);
    mapped_feats = Ks2 * D * (inv(V))^0.5;
end