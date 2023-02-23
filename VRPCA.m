function [error] = VRPCA(X,numepochs,winit,U_star,eps)
[d,n] = size(X);
m = n; % epoch length
eta = 1/(mean(sum(X.^2,1))*sqrt(n)); % step size
error = zeros(numepochs/2,1);

% initialize w
if nargin<4
    winit = randn(d,1); 
end
w = winit/norm(winit);
wrun = w;

for i = 1 : floor(numepochs/2)
    error(i) = 1 - norm(w'*U_star)^2/norm(w)^2;
    if error(i) < eps
        break;
    end
    u = (1/n)*X*(X'*w);
    for j=1:m
        s = randi(n);
        wrun = wrun+eta*(((X(:,s)'*(wrun-w)))*X(:,s)+u);
        wrun = wrun/norm(wrun);
    end
    w = wrun;
end
