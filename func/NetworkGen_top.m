function [W,neighbor,neigh_num] = NetworkGen_top(Nv)

networkRadius = 10;
[cmat, incimat, nnum, Coordinates] = NetworkGen(Nv, 50, 50, networkRadius);
incimat2 = incimat(:,1:2:end);
V = incimat2';
[N_eg, N_nd] = size(V);
L = V' * V; % the diagonal is the degree, (i,j) entry indicates whether there is an edge by '-1' and '0'.
tau = max(diag(2*L));
W = eye(Nv)-2*L/tau; % sum of each row and each column is one
if Nv==1
    W = 1;
end
neighbor = cell(Nv,1);
neigh_num = zeros(Nv,1);
for n = 1 : Nv
    neighbor{n} = find(W(n,:)>0);
    neighbor{n}(neighbor{n}==n) = [];
    neigh_num(n) = length(neighbor{n});
end
end