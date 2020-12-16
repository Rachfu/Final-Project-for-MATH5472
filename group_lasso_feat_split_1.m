function [x, history] = group_lasso_feat_split_1(A, b, lambda, ni, RHO, ALPHA, iteration)
t_start = tic;
QUIET    = 0;
MAX_ITER = iteration;
RELTOL  = 1e-2;
ABSTOL   = 1e-4;[m, n] = size(A);

% check that ni divides in to n
if (rem(n,ni) ~= 0)
    error('invalid block size');
end
% number of subsystems
N = n/ni;
rho = RHO;
alpha = ALPHA;    % over-relaxation parameter

x = zeros(ni,N);
z = zeros(m,1);
u = zeros(m,1);
Axbar = zeros(m,1);

zs = zeros(m,N);
Aixi = zeros(m,N);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

% pre-factor
for i = 1:N
    Ai = A(:,(i-1)*ni + 1:i*ni);
    [Vi,Di] = eig(Ai'*Ai);
    V{i} = Vi;
    D{i} = diag(Di);

    % in Matlab, transposing costs space and flops
    % so we save a transpose operation everytime
    At{i} = Ai';
end

for k = 1:MAX_ITER
    % x-update (to be done in parallel)
    for i = 1:N
        Ai = A(:,(i-1)*ni + 1:i*ni);
        xx = x_update(Ai, Aixi(:,i) + z - Axbar - u, lambda/rho, V{i}, D{i});
        x(:,i) = xx;
        Aixi(:,i) = Ai*x(:,i);
    end

    % z-update
    zold = z;
    Axbar = 1/N*A*vec(x);

    Axbar_hat = alpha*Axbar + (1-alpha)*zold;
    z = (b + rho*(Axbar_hat + u))/(N+rho);

    % u-update
    u = u + Axbar_hat - z;

    % compute the dual residual norm square
    s = 0; q = 0;
    zsold = zs;
    zs = z*ones(1,N) + Aixi - Axbar*ones(1,N);
    for i = 1:N
        % dual residual norm square
        s = s + norm(-rho*At{i}*(zs(:,i) - zsold(:,i)))^2;
        % dual residual epsilon
        q = q + norm(rho*At{i}*u)^2;
    end

    % diagnostics, reporting, termination checks
    history.objval(k)  = objective_1(A, b, lambda, N, x, z);
    history.r_norm(k)  = sqrt(N)*norm(z - Axbar);
    history.s_norm(k)  = sqrt(s);

    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(Aixi,'fro'), norm(-zs, 'fro'));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*sqrt(q);
    
    sum = 0;
    x_r = x(:,1)';
    
    for m = 1:N-1
        x_r = [x_r x(:,m+1)'];
    end
    x_r = x_r';
    for m = 1:N
        sum = sum + norm(x(:,m));
    end
    
    history.suboptimality(k) = norm(A*x_r-b)^2/2+lambda*sum;


    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

end

if ~QUIET
    toc(t_start);
end
end