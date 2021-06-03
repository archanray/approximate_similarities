% Code to compute nystrom on top of WMD features
% reuses the WMD library made available by  Lingfei Wu
%
% Author: Archan Ray
% Date: 19 October, 2020

function[mapped_feats, samples, sample_weights, user_emd_time] = WMD_CUR(train_X,...
    train_weight_X, gamma, sample_size, samples, sample_weights)
    
    % reduce set here before using Nystrom
    n = size(train_X,2);
    if size(samples, 1) == 0
        % train case
        s = randsample(1:n, sample_size);
    end
    % support samples
    if size(samples, 1) == 0 
        % train case
        samples = train_X(:, s);
        sample_weights = train_weight_X(:, s);
    end
    % compute the respective WMD values
    [C, C_emd_time] = wmd_dist(train_X,train_weight_X,...
        samples,sample_weights,gamma);
    C = C / (((1/n)*sample_size)^0.5);
    if size(samples, 1) == 0
        psi = C(s, :);
        psi_emd_time = 0;
    else
        [psi, psi_emd_time] = wmd_dist(samples,sample_weights,...
        samples,sample_weights,gamma);
    end
    psi = psi / (((1/n)*sample_size)^0.5);
    
    % house cleaning
    C(C > 1e+08) = 0;
    psi(psi > 1e+08) = 0;
    
    U = C'*C;
    U = eps*eye(size(U,1));
    U = U \ psi;
    
    
    [S,V,~] = svd(U);
    sqrtU = S*(V^0.5);
    
    user_emd_time = C_emd_time+psi_emd_time;
    
    % computing features
    mapped_feats = C * sqrtU;
end