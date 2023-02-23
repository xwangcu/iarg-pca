function [error] = VRPowerM(X,numepochs,w,U_star,lambda2,eps) 

n = size(X,2);
beta = lambda2^2/4;
m = n;
s = 1;
w = w / norm(w);
w0 = 0;
error = zeros(numepochs/2,1);
A = (X*X')/n;
w_tilde = w;

for k = 1 : floor(numepochs/2)
    error(k) = 1 - norm(w'*U_star)^2/norm(w)^2;
    if error(k) < eps
        break;
    end
    v_tilde = A * w_tilde;
    for t = 1 : m
        alpha = (w'*w_tilde);
        id = randperm(n,s);
        w0 = w;
        T = X(:,id) * ( X(:,id)' * (w-alpha*w_tilde) );
        w = T/s + alpha*v_tilde - beta*w0;
        z = norm(w);
        w0 = w0/z;
        w = w/z;
    end
    w_tilde = w;
end
