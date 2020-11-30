function error = calculateJ(X)
    error = 0;
    for i=1:size(X,2)
        for j=1:size(X,2)
            error = error + sum((X(:,i) - X(:,j)).^2, 1);
        end
    end
    error = (1 / (2*size(X,2))) * error;
end