function [clust1, clust2] = spectral_kmeans(X)
    clust1 = [];
    clust2 = [];
    if size(X, 2) <= 1
        return
    end
    D = zeros(1, size(X, 2));
    W = zeros(size(X,2));
    for i=1:size(X, 2)
        temp = 0;
        for j=1:size(X,2)
            dist = exp( -1.*sum((X(:,i) - X(:,j)).^2, 1) / 10);
            if i == j
                dist = 0;
            end
            W(i, j) = dist;
            temp = temp + dist;
        end
        D(1, i) = temp;
    end
    D = diag(D);
    diff = D - W;
    mat = inv(D)^0.5 * (diff) * inv(D)^0.5;
    [V,D] = eig(mat);
    [d,ind] = sort(diag(D));
    Ds = D(ind,ind);
    Vs = V(:,ind);
    V_imp = inv(D)^0.5 * Vs(:, 2);
    for i=1:size(V_imp, 1)
        if V_imp(i) >= 0
            clust1 = [clust1, X(:,i)];
        else
            clust2 = [clust2, X(:,i)];
        end
    end
end