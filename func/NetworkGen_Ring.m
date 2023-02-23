function [W] = NetworkGen_Ring(N)

E = eye(N);
index = [N, (1:N)];
W = E(:, index);

end