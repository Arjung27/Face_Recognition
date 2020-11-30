function all_clusters = parallel_cust_kmeans(X, k)
    rng(k);
    all_clusters = containers.Map;
    minX = min(X(1,:));
    minY = min(X(2,:));
    maxX = max(X(1,:));
    maxY = max(X(2,:));
    mins = [minX; minY];
    maxs = [maxX; maxY];
    init_centers = (maxs - mins).*rand(2,k) + mins;
    n_samples = size(X, 2);
    error = 1e+06;
    while error > 1e-04
        all_clusters_temp = containers.Map;
        for j=1:n_samples
            dist1 = sum((X(:,j) - init_centers).^2, 1).^0.5;
            [M,I] = min(dist1);
            key = int2str(I);
            if isKey(all_clusters_temp, key)
                all_clusters_temp(key) = [all_clusters_temp(key), X(:,j)];
            else
                all_clusters_temp(key) = X(:,j);
            end
        end
            
        prev_centers = init_centers;
        init_centers = [];
        for l=1:k
            if isKey(all_clusters_temp, int2str(l))
                init_centers = [init_centers, mean(all_clusters_temp(int2str(l)), 2)];
                all_clusters(int2str(l)) = all_clusters_temp(int2str(l));
            else
                init_centers = [init_centers, [0;0]];
            end
        end
        error = sum(sum((prev_centers - init_centers).^2).^0.5);
    end
end