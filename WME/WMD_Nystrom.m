% Code to compute nystrom on top of WMD features
% reuses the WMD library made available by  Lingfei Wu
%
% Author: Archan Ray
% Date: 19 October, 2020

function[mapped_feats, samples, sample_weights, user_emd_time] = WMD_Nystrom(train_X,...
    train_weight_X, gamma, sample_size, R, samples, sample_weights, c_for_eig)
    
    % reduce set here before using Nystrom
    n = size(train_X,2);
    if size(samples, 1) == 0
        % train case
        s = randsample(1:n, sample_size);
        z = randsample(1:n, ceil((sample_size*n)^0.5));
    else
        % test case
        z = randsample(1:n, ceil(size(samples,2)^0.5));
    end
    % support samples
    if size(samples, 1) == 0 
        % train case
        samples = train_X(:, s);
        sample_weights = train_weight_X(:, s);
    end
    % compute the respective WMD values
    [Ks, Ks_emd_time] = wmd_dist(train_X,train_weight_X,...
        samples,sample_weights,gamma);
    [Kz, Kz_emd_time] = wmd_dist(train_X(:,z),train_weight_X(:,z),...
        train_X(:, z),train_weight_X(:, z),gamma);
    
    if size(samples, 1) > 0
        % test case
        [sKs, sKs_emd_time] = wmd_dist(samples, sample_weights,...
            samples, sample_weights, gamma);
    end
    
    user_emd_time = Ks_emd_time+Kz_emd_time;
    
    % consistency wih original code
    Ks = Ks / sqrt(R);
    Kz = Kz / sqrt(R);
    
    % house cleaning
    Ks(Ks > 1e+08) = 0;
    Kz(Kz > 1e+08) = 0;
    
    % compute the Nystrom approximation feature
    if size(samples, 1) == 0
        % train case
        sKs = Ks(s, :);
    end
    [Vs, Es] = eig(Kz);
    minEig = eigs(Kz, 1, 'smallestreal');
    minEigI = c_for_eig* minEig * eye(n,n);
    if size(samples, 1) == 0
        % train case
        Ks = Ks - minEigI(:,s);
        sKs = sKs - minEigI(s,s);
    else
        % test case
        sample_size = size(samples,2);
        sKs = sKs - minEig * eye(sample_size, sample_size);
    end
    isKs = inv(sKs);
    mapped_feats = Ks * isKs^0.5;
end