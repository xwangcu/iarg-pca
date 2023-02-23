function [error] = Oja(X,numepochs,winit,U_star,eta,eps)

n = size(X,2);
b = 1;
max_iter = numepochs*n;
error = zeros(numepochs/2,1);
w = winit;
for iter = 1 : max_iter
    if mod(iter,2*n) == 1
        error(floor(iter/(2*n))+1) = 1 - norm(w'*U_star)^2/norm(w)^2;
        if error(floor(iter/(2*n))+1) < eps
            break;
        end
    end
    id = randperm(n,b);
    prod = X(:,id)'*w;
    g = X(:,id)*prod;
    eta = eta/ceil(iter/n);
    w = w + eta*g;
    w = w / norm(w);
end
