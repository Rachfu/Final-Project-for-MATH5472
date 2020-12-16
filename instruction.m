%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Small Dense Lasso %%%%
% using functions by corresponding m files:
% objective.m; shrinkage.m; factor.m; lasso.m; lasso_1.m; lasso_2.m;
% lasso_3.m

% Data Generation
m = 1500;       % number of examples
n = 5000;       % number of features
A = randn(m,n);
A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n); % normalize columns
x0 = randn(n,1);
v = sqrt(0.001)*randn(m,1);
b = A*x0 + v;

lambda_max = norm( A'*b, 'inf' );
lambda = 0.1*lambda_max;

% solve Problem
[x history] = lasso(A, b, lambda, 1.0, 1.0);

% result
K = length(history.objval);
g = figure;
subplot(2,1,1);
semilogy(1:K, max(1e-8, history.r_norm), 'k', ...
    1:K, history.eps_pri, 'k--',  'LineWidth', 2);
ylabel('||r||_2');

subplot(2,1,2);
semilogy(1:K, max(1e-8, history.s_norm), 'k', ...
    1:K, history.eps_dual, 'k--', 'LineWidth', 2);
ylabel('||s||_2'); xlabel('iter (k)');

%%% iteration set to 35 %%%
p_star = norm(A*x0-b)^2/2+lambda*norm(x0,1);
[x history] = lasso_1(A, b, lambda, 1.0, 1.0, p_star);
K = length(history.objval);
g = figure;
subplot(2,1,1);
semilogy(1:K, max(1e-8, history.r_norm), 'k', ...
    1:K, history.eps_pri, 'k--',  'LineWidth', 2);
ylabel('||r||_2');

subplot(2,1,2);
semilogy(1:K, max(1e-8, history.s_norm), 'k', ...
    1:K, history.eps_dual, 'k--', 'LineWidth', 2);
ylabel('||s||_2'); xlabel('iteraion k');

%%% objective suboptimality %%%
semilogy(1:K, history.suboptimality, 'k', 'LineWidth', 2);
xlabel('iteration k'); a = ylabel('$$\tilde{p}^k-p^{*}$$');
set(a,'interpreter','latex');

%%% iteration needed for different lambda %%%
lambda_test = (0.01*lambda_max:(0.95*lambda_max-0.01*lambda_max)/99:0.95*lambda_max);
step_test = zeros(100,1);

% warmstart 
tic
[step_test(1),x,z,u] = lasso_2(A, b, lambda_test(1),1.0, 1.0, zeros(n,1),zeros(n,1),zeros(n,1));
for k = 2:100
    [step_test(k),x,z,u] = lasso_2(A, b, lambda_test(k),1.0, 1.0,x,z,u);
end
step_warmstart = step_test;
toc

% coldstart 
tic
for k = 1:100
    step_test(k) = lasso_3(A, b, lambda_test(k),1.0, 1.0);
end
step_coldstart = step_test;
toc

% comparision of warmstart and coldstart 
plot(step_coldstart,'k')
hold on
plot(step_warmstart,'k--')
xlabel('\lambda/\lambda_{max}')
ylabel('iterations needed')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Group Lasso with Feature Splitting %%%%
% using functions by corresponding m files:
% download package CVX at http://cvxr.com/cvx/download/ to use function vec
% objective_1.m; x_update.m; group_lasso_feat_split.m;
% group_lasso_feat_split_1.m

% data generation
m = 200;        % amount of data
K = 200;        % number of blocks
ni = 100;       % size of each block

n = ni*K;
p = 10/K;      % sparsity density

x = zeros(ni,K); % generate block sparse solution vector
for i = 1:K,
    if( rand() < p)
        % fill nonzeros
        x(:,i) = randn(ni,1);
    end
end
x = vec(x);

A = randn(m,n); % generate random data matrix

A = A*spdiags(1./norms(A)',0,n,n); % normalize columns of A

b = A*x + sqrt(1)*randn(m,1); % generate measurement b with noise

for i = 1:K,
    Ai = A(:,(i-1)*ni + 1:i*ni);
    nrmAitb(i) = norm(Ai'*b);
end
lambda_max = max( nrmAitb ); 

lambda = 0.5*lambda_max; % regularization parameter

xtrue = x;   % save solution

% solve problem

[x history] = group_lasso_feat_split(A, b, lambda, ni, 10, 1.0);

% result
K_1 = length(history.objval);

g = figure;
subplot(2,1,1);
semilogy(1:K_1, max(1e-8, history.r_norm), 'k', ...
    1:K_1, history.eps_pri, 'k--',  'LineWidth', 2);
ylabel('||r||_2');

subplot(2,1,2);
semilogy(1:K_1, max(1e-8, history.s_norm), 'k', ...
    1:K_1, history.eps_dual, 'k--', 'LineWidth', 2);
ylabel('||s||_2'); xlabel('iter (k)');

%%% iteration set to 100 %%%
lambda = 0.5*lambda_max;

xtrue = x;   % save solution

[x history] = group_lasso_feat_split_1(A, b, lambda, ni, 10, 1.0, 100);

K = length(history.objval);

g = figure;
subplot(2,1,1);
semilogy(1:K, max(1e-8, history.r_norm), 'k', ...
    1:K, history.eps_pri, 'k--',  'LineWidth', 2);
ylabel('||r||_2');

subplot(2,1,2);
semilogy(1:K, max(1e-8, history.s_norm), 'k', ...
    1:K, history.eps_dual, 'k--', 'LineWidth', 2);
ylabel('||s||_2'); xlabel('iter (k)');

%%% objective suboptimality %%%
[x, history] = group_lasso_feat_split_1(A, b, lambda, ni, 10, 1.0, 1000);
p_star = history.suboptimality(1000);
[x history] = group_lasso_feat_split_1(A, b, lambda, ni, 10, 1.0,100);
p_tilde = history.suboptimality;

p = p_tilde-p_star;

a = K_1*ones(10000,1);
b = linspace(0.01,100,10000);

semilogy(1:K, p, 'k', 'LineWidth', 2);
hold on
plot(a,b,'k--','LineWidth',2)
xlabel('iteration k'); a = ylabel('$$\tilde{p}^k-p^{*}$$');
set(a,'interpreter','latex');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Regressor Selection %%%%
% using functions by corresponding m files:
% objective_2.m; keep_largest.m; regressor_sel.m

% data generation
m = 1500;       % number of examples
n = 5000;       % number of features
p = 100/n;      % sparsity density

x = sprandn(n,1,p); % generate sparse solution vector

A = randn(m,n); % generate random data matrix

A = A*spdiags(1./sqrt(sum(A.^2))', 0, n, n); % normalize columns of A

b = A*x + sqrt(0.001)*randn(m,1); % generate measurement b with noise

xtrue = x;   % save solution

% solve problem
[x history] = regressor_sel(A, b, p*n, 1.0);

% result
K = length(history.objval);
subplot(2,1,1);
semilogy(1:K, max(1e-8, history.r_norm), 'k', ...
    1:K, history.eps_pri, 'k--',  'LineWidth', 2);
ylabel('||r||_2');

subplot(2,1,2);
semilogy(1:K, max(1e-8, history.s_norm), 'k', ...
    1:K, history.eps_dual, 'k--', 'LineWidth', 2);
ylabel('||s||_2'); xlabel('iter (k)');

% Illustration: some of my idea of the codes origins from the website of Stephen P. Boyd
