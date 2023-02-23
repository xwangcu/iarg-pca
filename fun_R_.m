
function [R] = fun_R(z,M,B,N,X_m,eta)

% U = reshape(z,[M,N]);
% R = zeros(M,N);
% for i = 1 : N
%     H = U(:,i) + eta * ( Cov(:,:,i)*U(:,i) - U(:,i) * (U(:,i)'*Cov(:,:,i)*U(:,i)) );
%     R(:,i) = H - U(:,i);
% end
% 
% R = reshape(R,[M*N,1]);

% R = zeros(M,N);
% for n = 1 : N
%    R(:,n) = z(:,n) + eta * ( Cov(:,:,n)*z(:,n) - z(:,n)*(z(:,n)'*Cov(:,:,n)*z(:,n)) ) - z(:,n);
% end

R = zeros(M,N);
for n = 1 : N
   prod = X_m(:,:,n)' * z(:,n);
   R(:,n) = z(:,n) + eta * (1/B) * ( X_m(:,:,n)*prod - z(:,n)*(norm(prod)^2) ) - z(:,n);
end

end

