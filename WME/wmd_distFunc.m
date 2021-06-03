function Ksub = wmd_distFunc(X, weight_X, rowInd, colInd, gamma)
    % this function computes the wmd distance between points
    if (isempty(colInd))
        Ksub = ones(length(rowInd),1);
    else
        % Ksub = zeros(length(rowInd), length(colInd));
        newX = cell(1, length(rowInd));
        weight_newX = cell(1, length(rowInd));
        baseX = cell(1, length(colInd));
        weight_baseX = cell(1, length(colInd));
        % get the rows of data points needed
        parfor i = 1:length(rowInd)
            newX{i} = X{i};
            weight_newX{i} = weight_X{i};
        end
        % get the samples
        parfor i = 1:length(colInd)
            baseX{i} = X{i};
            weight_baseX{i} = weight_X{i};
        end
        % compute the distance
        [Ksub,~] = wmd_dist(newX, weight_newX, baseX, weight_baseX, gamma);
        % return matrix
        % Ksub = exp(-gamma*Ksub);
    end

end