function [z, history] = lasso_1(A, b, lambda, rho, alpha, p_star)
t_start = tic;
RELTOL   = 1e-2;

[m, n] = size(A);

% save a matrix-vector multiply

QUIET    = 0;
MAX_ITER = 35;
ABSTOL   = 1e-4;
Atb = A'*b;

x = zeros(n,1);
z = zeros(n,1);
u = zeros(n,1);

% cache the factorization
[L U] = factor(A, rho);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective','suboptimality');
end

for k = 1:MAX_ITER

    % x-update
    q = Atb + rho*(z - u);    % temporary value
    if( m >= n )    % if skinny
       x = U \ (L \ q);
    else            % if fat
       x = q/rho - (A'*(U \ ( L \ (A*q) )))/rho^2;
    end

    % z-update with relaxation
    zold = z;
    x_hat = alpha*x + (1 - alpha)*zold;
    z = shrinkage(x_hat + u, lambda/rho);

    % u-update
    u = u + (x_hat - z);

    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(A, b, lambda, x, z);

    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - zold));

    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);
    
    history.suboptimality_1(k) = norm(A*z-b)^2/2+lambda*norm(z,1)-p_star;
    history.suboptimality_2(k) = norm(A*x-b)^2/2+lambda*norm(x,1)-p_star;
    
    history.suboptimality(k) = max(history.suboptimality_1(k),history.suboptimality_2(k));

    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k)), history.suboptimality;
    end

end

if ~QUIET
    toc(t_start);
end
end
