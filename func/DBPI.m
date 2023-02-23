%initialization for DBPI
Cov = zeros(M,M,N);
for n = 1 : N
    Cov(:,:,n) = sparse((1/B) * X_m(:,:,n) * (X_m(:,:,n))');
end
z = u0*ones(1,N);
err_dbpi = zeros(max_iter,1);
z0 = u0*ones(1,N); z1 = u0*ones(1,N);
alpha= 0.5; eta = 0.1;
for n = 1 : N
   z1(:,n) = z0 * W(:,n) + alpha*fun_R(z0,Cov,eta);
end

for n = 1 : N
    err_temp = 1 - norm(z(:,n)'*U_star)^2/norm(z(:,n))^2;
    err_dbpi(iter) = err_dbpi(iter) + err_temp/N;
end

Rz0 = fun_R(z0,M,N,Cov,eta);
Rz1 = fun_R(z1,M,N,Cov,eta);
for n = 1 : N   
   z(:,n) = z1(:,n) + z*W(:,n) - 0.5*z0(:,n) - 0.5*z0*W(:,n) + alpha*(Rz1(:,n)-Rz0(:,n));
end
z0 = z1;
z1 = z;



