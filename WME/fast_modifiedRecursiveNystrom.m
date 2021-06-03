function [C,W] = fast_modifiedRecursiveNystrom(trainData,gamma, sample_size, c_for_eig, accelerated_flag)
    
    
    %% Parameter processing and defaults    
    if nargin == 0
         error('recursiveNystrom:TooFewInputs','requires at least 1 input argument');
    end
    if nargin < 5
        accelerated_flag = 0;
    end
    % kernelFunc and s parameters should really be set by the user but we 
    % provide some defaults
    
    s = samples_size;
    
    n = size(trainData,1); % data size

    if(~accelerated_flag)
        % in the standard algorithm s samples are used in the final Nystrom
        % approximation as well as at each recursive level
        sLevel = s;
    else
        % in the accelerated version  < s samples are used at recursive 
        % levels to keep the total runtime at O(n*s)
        sLevel = ceil(sqrt((n*s + s^3)/(4*n)));
    end
    
    %% Start of algorithm  
    oversamp = log(sLevel);
    k = ceil(sLevel/(4*oversamp));
    nLevels = ceil(log(n/sLevel)/log(2));
    % random permutation for successful uniform samples
    perm = randperm(n);
    
    % set up sizes for recursive levels
    lSize = zeros(1,nLevels+1);
    lSize(1) = n;
    for i = 2:nLevels+1
        lSize(i) = ceil(lSize(i-1)/2);
    end
    
    % rInd: indices of points selected at previous level of recursion 
    % at the base level it's just a uniform sample of ~sLevel points
    samp = 1:lSize(end);
    rInd = perm(samp);
    weights = ones(length(rInd),1);

    % we need the diagonal of the whole kernel matrix, so compute upfront
    kDiag = diag(trainData(:, 2:end));
    
    %% Main recursion, unrolled for efficiency 
    for l = nLevels:-1:1  
        % indices of current uniform sample
        rIndCurr = perm(1:lSize(l));
        % build sampled kernel
        KS = trainData
        KS = kernelFunc(X,rIndCurr,rInd);
        SKS = KS(samp,:);
        SKSn = size(SKS,1);
        
        % optimal lambda for taking O(klogk) samples
        if(k >= SKSn)
            % for the rare chance we take less than k samples in a round
            lambda = 10e-6;
            % don't set to exactly 0 to avoid stability issues
        else
            lambda = (sum(diag(SKS).*weights.^2) - sum(abs(real(eigs(@(x) (SKS*(x.*weights)).*weights, SKSn, k)))))/k;
        end
        
        % compute and sample by lambda ridge leverage scores
        if(l ~= 1)
            % on intermediate levels, we independently sample each column
            % by its leverage score. the sample size is sLevel in expectation
            R = inv(SKS + diag(lambda*weights.^(-2)));
            % max(0,.) helps avoid numerical issues, unnecessary in theory
            levs = min(1,oversamp*(1/lambda)*max(0,(kDiag(rIndCurr) - sum((KS*R).*KS,2))));
            samp = find(rand(1,lSize(l)) < levs');
            % with very low probability, we could accidentally sample no
            % columns. In this case, just take a fixed size uniform sample.
            if(isempty(samp))
                levs(:) = sLevel/lSize(l);
                samp = randperm(lSize(l),sLevel);
            end
            weights = sqrt(1./(levs(samp)));

        else
            % on the top level, we sample exactly s landmark points without replacement
            R = inv(SKS + diag(lambda*weights.^(-2)));
            levs = min(1,(1/lambda)*max(0,(kDiag(rIndCurr) - sum((KS*R).*KS,2))));
            samp = datasample(1:n,s,'Replace',false,'Weights',levs);
        end
        rInd = perm(samp);
    end

    % build final Nystrom approximation
    % pinv or inversion with slight regularization helps stability
    C = kernelFunc(X,1:n,rInd);
    SKS = C(rInd,:);
    W = inv(SKS+(10e-6)*eye(s,s));
end
