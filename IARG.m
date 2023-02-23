function [error] = IARG(X_m,numepochs,winit,U_star,eta,eps) 
[d,n] = size(X_m);
u_IARG = winit; 
U_m = zeros(d,n); 
max_iter = numepochs*n;
error = zeros(numepochs/2,1); 
grad = zeros(d,1);
X = X_m; 
indx = 1; 
G = @(x,u)(x*(x'*u) - u*(u'*x*x'*u));

for iter = 1 : max_iter
    if mod(iter,2*n) == 1
        error(floor(iter/(2*n))+1) = 1 - norm(u_IARG'*U_star)^2/norm(u_IARG)^2;
        if error(floor(iter/(2*n))+1) < eps
            break;
        end
    end
    grad = grad + 1/n * ( G(X_m(:,indx),u_IARG) - G(X(:,indx),U_m(:,indx)) );
    U_m(:,indx) = u_IARG;
    u_IARG = u_IARG + eta*grad;
    indx = mod(indx, n) + 1;
end