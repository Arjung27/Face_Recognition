data_copy = load("hw06-data1.mat");
X = data_copy.X;

figure(1)
plot(X(1,:), X(2, :),'*')
title('Original data')
saveas(gca, 'que2_original_data.png')

x_data = X(1,:)';
y_data = X(2,:)';

n_samples = size(X,2);
x_new = x_data - mean(x_data).*ones(n_samples, 1);
y_new = y_data - mean(y_data).*ones(n_samples, 1);
cov_matrix = cov(x_new, y_new);
[~, ~, V] = svd(cov_matrix);  
final_data1 = V'*[x_new, y_new]';
figure(2)
plot(final_data1(1,:), final_data1(2, :),'^')
title('Data points projected over first linear principal component')
saveas(gca, 'que2_first_linear_pca.png')

figure(3)
histogram(final_data1, 'DisplayStyle','bar')
title('Histogram of the projected data - Linear PCA')
saveas(gca, 'que2_projected_linear.png')

data_copy = X;
dim = 2;
K = zeros(size(data_copy, 2),size(data_copy, 2));
for row = 1:size(data_copy, 2)
    for col = 1:row
        temp = sum(((data_copy(:,row) - data_copy(:,col)).^2));
        K(row,col) = exp(-temp / 50); 
    end
end

K = K + K';
for row = 1:size(data_copy, 2)
    K(row,row) = K(row,row)/2;
end

K_centered = K - ones(size(K))*K - K*ones(size(K)) + ones(size(K))*K*ones(size(K));
clear K

neighbours = 30;
opts.issym=1;                          
opts.disp = 0; 
opts.isreal = 1;
[eig_vec, eig_val] = eigs(K_centered,[],neighbours,'lm',opts);
eig_val = eig_val ~= 0;
eig_val = eig_val./size(data_copy,2);

for col = 1:size(eig_vec,2)
    eig_vec(:,col) = eig_vec(:,col)./(sqrt(eig_val(col,col)));
end
[~, index] = sort(eig_val,'descend');
eig_vec = eig_vec(:,index);


final_data = zeros(dim,size(data_copy,2));
for count = 1:dim
    final_data(count,:) = eig_vec(:,count)'*K_centered';
end

figure(4)
plot(final_data(1,:), final_data(2, :),'s')
title('Data points projected over first kernel principal component')
saveas(gca, 'que2_first_kernel_pca.png')

figure(5)
histogram(final_data, 'DisplayStyle','bar')
title('Histogram of the projected data - Kernel PCA')
saveas(gca, 'que2_projected_kernel_pca.png')