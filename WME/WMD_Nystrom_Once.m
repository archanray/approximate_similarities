% Code to compute nystrom on top of WMD features
% reuses the WMD library made available by  Lingfei Wu
%
% Author: Archan Ray
% Date: 19 October, 2020

function[mapped_feats] = WMD_Nystrom_Once(X, gamma, sample_size, chosen_samples, c)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    n = size(X,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % get samples to be considered
    available_samples = [1:length(chosen_samples)];
    samples_considered = datasample(available_samples, sample_size, 'Replace',false);
    large_samples_considered = datasample(available_samples, ...
                    ceil((sample_size*length(X))^0.5), 'Replace',false);
    samples_considered_indices = chosen_samples(1, samples_considered);
    large_samples_considered_indices = chosen_samples(1, large_samples_considered);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % subsample the original matrix
    new_X = X.^gamma;
    reduced_X = new_X(samples_considered_indices, samples_considered);
    large_reduced_X = new_X(large_samples_considered_indices, large_samples_considered);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % eigenvalue correction
    [~, Es] = eig(large_reduced_X);
    minEig = min(0, min(diag(Es))) - 0.001;
    minEigI = c * minEig * eye(n,n);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % compute the approximate features as Kz
    Ks = new_X(:, samples_considered) - minEigI(:, samples_considered);
    sKs = reduced_X - minEigI(samples_considered, samples_considered);
    isKs = inv(sKs);
    mapped_feats = Ks * isKs^0.5;
end
