K = zeros(10);
I = eye(10);
one_mat = ones(10,1);
gamma = 0.1;
lambda = 0.01;
N = 10;
x1 = x';
y1 = y';

for n=1:10
    for m=1:10
        K(m,n) = exp(-gamma*(x1(m)-y1(n))^2);
    end
end

A = [(K+lambda*I), one_mat; one_mat'*K, N];
B = [y1; one_mat'*y1];
solver = linsolve(A,B);
intervals = linspace(-2,2);

for i = 1:100
    output(i) = solver(11);
    for j = 1:10
        output(i) = output(i) + solver(j)*exp(-gamma*(intervals(i)-x1(j))^2);
    end
end

figure
hold on
plot(x1, y1, 'o')
plot(intervals, output)
saveas(gcf, 'value1')
