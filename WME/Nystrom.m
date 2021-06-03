function [reduced_set, error] = Nystrom(similarity_matrix, sample_size)

    K = similarity_matrix;
    % convert to symmetric
    K = (K+K') / 2;
%     chol(K);
    [V,E] = eig(K);
    n = size(K,1);
    
%     error = 0;
    
    s = randsample(1:n, sample_size);
    z = randsample(1:n, ceil((sample_size*n)^0.5));
    [Vs, Es] = eig(K(z,z));
    minEig = min(0, min(diag(Es))) - 0.001;
    Kbar = K - minEig*eye(n,n);
    error = norm(K - Kbar(:,s)*inv(Kbar(s,s))*Kbar(s, :)) / norm(K);
    
    % reduced_set = Kbar(:,s) * inv(Kbar(s,s))^0.5 ; 
    reduced_set = Kbar(:,s) * inv(Kbar(s,s))^0.5 ; 
end