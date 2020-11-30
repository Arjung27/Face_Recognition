data1 = load('hw06-data1.mat');
data2 = load('hw06-data2.mat');
X1 = data1.X;
X2 = data2.X;
for j=1:3
    k = power(2, j);
    [clust1, clust2] = spectral_kmeans(X2);
    [clust3, clust4] = spectral_kmeans(clust1);
    [clust5, clust6] = spectral_kmeans(clust2);
    [clust7, clust8] = spectral_kmeans(clust3);
    [clust9, clust10] = spectral_kmeans(clust4);
    [clust11, clust12] = spectral_kmeans(clust5);
    [clust13, clust14] = spectral_kmeans(clust6);
    markers = ['o', 'x', '*', 's', 'p', 'h', '+', '^'];
    map = containers.Map;
    total_cost = 0;
    if k == 8
        map('1') = clust7
        map('2') = clust8;
        map('3') = clust9;
        map('4') = clust10;
        map('5') = clust11;
        map('6') = clust12;
        map('7') = clust13;
        map('8') = clust14;
         
    elseif k == 4
        map('1') = clust3;
        map('2') = clust4;
        map('3') = clust5;
        map('4') = clust6;            
        
    elseif k ==2
        map('1') = clust1;
        map('2') = clust2;           
    end
    figure
    hold on
    for i=1:k
         cluster_curr = map(int2str(i));
         plot(cluster_curr(1,:), cluster_curr(2,:), markers(i))
         total_cost = total_cost + calculateJ(cluster_curr);
    end
    saveas(gcf, strcat('que1b_X2_m=', int2str(k), '_totalcost=', num2str(total_cost, 2), '.png'))
    
end