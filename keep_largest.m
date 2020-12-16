function z = keep_largest(z, K)
    [val pos] = sort(abs(z), 'descend');
    z(pos(K+1:end)) = 0;
end