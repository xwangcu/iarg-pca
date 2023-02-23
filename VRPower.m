function [error] = VRPower(X,numepochs,winit,U_star,eta,eps)

n = size(X,2);
m = n;
b = 1;
error = zeros(numepochs/2,1);

C = (X*X')/n;
w_tilde = winit;
for s = 1 : floor(numepochs/2)
%     fprintf("epoch = %d\n",s);
    error(s) = 1 - norm(w_tilde'*U_star)^2/norm(w_tilde)^2;
    g_tilde = C * w_tilde;
    w0 = w_tilde;
    w = (1-eta)*w0 + eta*g_tilde;
    if error(s) < eps
        break;
    end
    for t = 1 : m
%         fprintf("t = %d\n",t);
        id = randperm(n,b);
%         A = X(:,id)*X(:,id)';
%         g = A*w - A*w0*((w0')*w)/norm(w0)^2 + (w')*w0*g_tilde/norm(w0)^2;
        g = X(:,id)*(X(:,id)'*w) - X(:,id)*(X(:,id)'*w0)*((w0')*w)/norm(w0)^2 + (w')*w0*g_tilde/norm(w0)^2;
        w = (1-eta)*w + eta*g;
    end
    w_tilde = w;
end
