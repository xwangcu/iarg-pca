clear all;
load cifar10.mat
X = Z';

N = 50; B = 1000; M =3072; K =1;
X = X - mean(X,2)*ones(1,size(X,2));
X_m = zeros(M,B,N);
for n = 1 : N
    X_m(:,:,n) = X(:,(n-1)*B+1:n*B);
end
[U,S,~] = svds(sparse(X));
U_star = U(:,1);
clear X Z;

% load topology_er_50.mat W
load topology_ring.mat W %mixi ng matrix
P = W; P(1:N+1:end)=0;
neighbor = cell(N,1);
for n = 1 : N
    % neighbors of the n-th agent
    neighbor{n} = find(P(:,n)>0);
end

u0 = orth(randn(M,1)); %initilization
Gr = @(x,u)(x*(x'*u) - u*(u'*x*x'*u));
G = @(x,u)(x*(x'*u));

%algorithmic parameters
max_iter = 2000;
mu_rw = 1E-11;%2E-11;%1E-4;%0.5E-4;
mu_kr = 2E-11*0.5;%0.7
mu_oja = 2E-11*0.5;%0.7
K_de = 20; %1 (ER,cifar), 20 (Ring,cifar);
s = svd(W); s = s(2);
eta = (1-sqrt(1-s^2))/(1+sqrt(1-s^2));


%initialization for random walk
u_rw = u0;  u_rwm = zeros(M,N); err_rw = zeros(max_iter,1);
grad_rw = zeros(M,K);

%initialization for Krasulina
u_kr = u0; err_kr = zeros(max_iter,1); grad_kr = zeros(M,1);

%initialization for Oja
u_oja = u0;  err_oja = zeros(max_iter,1); grad_oja = zeros(M,1);

%initialization for DeEPCA
u_de = u0*ones(1,N); err_de = zeros(max_iter,1); s_de = u_de;

% %initialization for DBPI
% Cov = zeros(M,M,N);
% for n = 1 : N
%     Cov(:,:,n) = sparse((1/B) * X_m(:,:,n) * (X_m(:,:,n))');
% end
% u_dbpi = sparse(u0*ones(1,N));
% err_dbpi = zeros(max_iter,1);
% W_kron = sparse(kron(W,sparse(eye(M))));
% u_dbpi0 = kron(ones(N,1),u0);
% alpha= 0.5; eta = 0.1;
% u_dbpi1 = W_kron*u_dbpi0 + alpha*fun_R(u_dbpi0,M,N,Cov,eta_dbpi);
%initialization for DBPI
% Cov = zeros(M,M,N);
% for n = 1 : N
%     Cov(:,:,n) = sparse((1/B) * X_m(:,:,n) * (X_m(:,:,n))');
% end

alpha= 0.5; 
eta_dbpi = 1e-7;
z = u0*ones(1,N);
z0 = u0*ones(1,N); 
z1 = u0*ones(1,N);
err_dbpi = zeros(max_iter,1);
Rz0 = fun_R(z0,M,B,N,X_m,eta_dbpi);
for n = 1 : N
    z1(:,n) = z0 * W(:,n) + alpha*Rz0(:,n);
end

indx_rw = 1; indx_rw_pre = 1; indx_kr = 1; indx_oja = 1;

for iter = 1 : max_iter
    fprintf('iter = %d\n', iter);

    %%    random walk
    disp('rw');
    err_rw(iter) = 1 - norm(u_rw'*U_star)^2/norm(u_rw)^2;
    grad_rw = grad_rw + 1/N*(Gr(X_m(:,:,indx_rw),u_rw) - Gr(X_m(:,:,indx_rw),u_rwm(:,indx_rw)));
    u_rwm(:,indx_rw) = u_rw;
    u_rw = u_rw + mu_rw * grad_rw;
    u_rw = u_rw/norm(u_rw);
    neighbor_temp = neighbor{indx_rw}; neighbor_temp(neighbor_temp==indx_rw_pre) = [];
    indx_temp = randi(length(neighbor_temp));
    indx_rw_pre = indx_rw;
    indx_rw = neighbor_temp(indx_temp);

    %% Krasulina
    disp('Krasulina');
    err_kr(iter) = 1 - norm(u_kr'*U_star)^2/norm(u_kr)^2;
    grad_kr = Gr(X_m(:,:,indx_kr),u_kr);
    u_kr = u_kr + mu_kr * grad_kr;
    u_kr = u_kr/norm(u_kr);
    indx_kr = mod(indx_kr, N) + 1;

    %% Oja
    disp('Oja');
    err_oja(iter) = 1 - norm(u_oja'*U_star)^2/norm(u_oja)^2;
    grad_oja = G(X_m(:,:,indx_oja),u_oja);
    u_oja = u_oja + mu_oja * grad_oja;
    u_oja = u_oja/norm(u_oja);
    indx_oja = mod(indx_oja, N) + 1;

    %% DeEPCA
    disp('DeEPCA');
    for n = 1 : N
        err_temp = 1 - norm(u_de(:,n)'*U_star)^2/norm(u_de(:,n))^2;
        err_de(iter) = err_de(iter) + err_temp/N;
    end
    if iter > 1
        u_de_temp(:,1,:) = u_de - u_de_pre;
        s_de = s_de + squeeze(pagemtimes(X_m,pagemtimes(X_m,'transpose',u_de_temp,'none')));
    else
        u_de_temp(:,1,:) = u_de;
        s_de = s_de + squeeze(pagemtimes(X_m,pagemtimes(X_m,'transpose',u_de_temp,'none'))) - u_de;
    end
    s_de_pre = s_de;
    for k = 1 : K_de
        s_de_new = (1+eta)*s_de*W - eta*s_de_pre;
        s_de_pre = s_de;
        s_de = s_de_new;
    end
    u_de_new = s_de./(ones(M,1)*sqrt(sum(s_de.^2,1)));
    for n = 1 : N
        if u_de_new(:,n)'*U_star<0
            u_de_new(:,n) = -u_de_new(:,n);
        end
    end
    u_de_pre = u_de;
    u_de = u_de_new;

    %% DBPI
    disp('DBPI');
    for n = 1 : N
        err_temp = 1 - norm(z(:,n)'*U_star)^2/norm(z(:,n))^2;
        err_dbpi(iter) = err_dbpi(iter) + err_temp/N;
    end

    Rz0 = fun_R(z0,M,B,N,X_m,eta_dbpi);
    Rz1 = fun_R(z1,M,B,N,X_m,eta_dbpi);
    for n = 1 : N
        z(:,n) = z1(:,n) + z1*W(:,n) - 0.5*z0(:,n) - 0.5*z0*W(:,n) + alpha*(Rz1(:,n)-Rz0(:,n));
    end
    z0 = z1;
    z1 = z;
end
err_rw = err_rw./(1-err_rw);
err_kr = err_kr./(1-err_kr);
err_oja = err_oja./(1-err_oja);
err_de = err_de./(1-err_de);
err_dbpi = err_dbpi./(1-err_dbpi);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot communication cost
figure;
hold on; 
box on;
color = get(gca,'colorOrder');
plot((0:max_iter-1),err_kr,'color','b','LineWidth',2.5);
plot((0:max_iter-1),err_oja,'color','#EDB120','LineWidth',1.5,'LineStyle', '-.');
plot((0:round(max_iter/N))*(sum(W(:)>0)-N)*K_de,err_de(1:round(max_iter/N)+1),'color','k','LineWidth',1.5,'LineStyle', '--');
plot((0:round(max_iter/N))*(sum(W(:)>0)-N),err_dbpi(1:round(max_iter/N)+1),'color','g','LineWidth',1.5);
plot((0:max_iter-1)*2,err_rw,'color','r','LineWidth',1.5);

set(gca,'yscale','log');
ax = gca;
ax.FontSize = 15;
ax.XAxis.Exponent = 3;
xlabel('Communication Cost','FontSize',20);
ylabel('{$1-\langle \textbf{{\emph w}}^t,\textbf{{\emph v}}_1 \rangle^2$}','Interpreter','latex','FontSize',25);
lgd = legend('Krasulina','Oja','DeEPCA','DBPI','IARG-PCA','location','southeast');
lgd.FontSize = 13;
title('Erdős–Rényi (CIFAR-10)');
xlim([1,20e3]);



% % plot iteration number
% figure;
% hold on; 
% box on;
% color = get(gca,'colorOrder');
% plot((0:max_iter-1),err_kr,'color','b','LineWidth',2.5);
% plot((0:max_iter-1),err_oja,'color','#EDB120','LineWidth',1.5,'LineStyle', '-.');
% plot((0:max_iter-1),err_de,'color','k','LineWidth',1.5,'LineStyle', '--');
% plot((0:max_iter-1),err_dbpi,'color','g','LineWidth',1.5);
% plot((0:max_iter-1),err_rw,'color','r','LineWidth',1.5);
% 
% set(gca,'yscale','log');
% ax = gca;
% ax.FontSize = 15;
% ax.XAxis.Exponent = 3;
% xlabel('Iteration Number','FontSize',20);
% ylabel('{$1-\langle \textbf{{\emph w}}^t,\textbf{{\emph v}}_1 \rangle^2$}','Interpreter','latex','FontSize',25);
% lgd = legend('Krasulina','Oja','DeEPCA','DBPI','IARG-PCA','location','southeast');
% lgd.FontSize = 13;
% xlim([2,1e3]);
% ylim([1e-12,1e1]);

