data1 = load('hw06-data1.mat');
data2 = load('hw06-data2.mat');
X1 = data1.X;
X2 = data2.X;

for j=1:3
    k = power(2, j);
    cluster = parallel_cust_kmeans(X2, k);
    markers = ['o', 'x', '*', 's', 'p', 'h', '+', '^'];
    total_cost = 0;
    figure
    hold on
     for i=1:k
         if isKey(cluster, int2str(i))
             cluster_curr = cluster(int2str(i));
             total_cost = total_cost + calculateJ(cluster_curr);
             plot(cluster_curr(1,:), cluster_curr(2,:), markers(i))
         end
     end
      saveas(gcf, strcat('que1a_X2_m=', int2str(k), '_totalcost=', num2str(total_cost, 2), '.png'))
end