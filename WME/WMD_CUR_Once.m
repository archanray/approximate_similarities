% Code to compute nystrom on top of WMD features
% reuses the WMD library made available by  Lingfei Wu
%
% Author: Archan Ray
% Date: 

function[mapped_feats] = WMD_CUR_Once(X, gamma, sample_size, chosen_samples)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    n = size(X,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % get samples to be considered
    available_samples = [1:length(chosen_samples)];
    samples_considered = datasample(available_samples, sample_size, 'Replace',false);
    samples_considered_indices = chosen_samples(1, samples_considered);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % subsample the original matrix
    new_X = X.^gamma;
    reduced_X = new_X(samples_considered_indices, samples_considered);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % compute the approximate features
    C = new_X(:, samples_considered) / (((1/n)*sample_size)^0.5);
    psi = reduced_X / (((1/n)*sample_size)^0.5);
    U = C' * C;
    U = U \ psi ;
    
    [S,V,~] = svd(U);
    sqrtU = S*(V^0.5);
    mapped_feats = C * sqrtU;
end
