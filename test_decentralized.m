clear all; 
%close all;
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

%load topology_er_50.mat W
load topology_ring.mat W %mixing matrix
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
max_iter = 2E3;
mu_rw = 2E-11;%1E-4;%0.5E-4;
mu_kr = mu_rw*0.5;%0.7
mu_oja = mu_rw*0.5;%0.7
K_de = 20;%1;%20;
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
indx_rw = 1; indx_rw_pre = 1; indx_kr = 1; indx_oja = 1;

for iter = 1 : max_iter
%%    random walk
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
    err_kr(iter) = 1 - norm(u_kr'*U_star)^2/norm(u_kr)^2;
    grad_kr = Gr(X_m(:,:,indx_kr),u_kr);
    u_kr = u_kr + mu_kr * grad_kr;
    u_kr = u_kr/norm(u_kr);
    indx_kr = mod(indx_kr, N) + 1;
   
%% Oja
    err_oja(iter) = 1 - norm(u_oja'*U_star)^2/norm(u_oja)^2;
    grad_oja = G(X_m(:,:,indx_oja),u_oja);
    u_oja = u_oja + mu_oja * grad_oja;
    u_oja = u_oja/norm(u_oja);
    indx_oja = mod(indx_oja, N) + 1;
    
%% DeEPCA
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
 
end
err_rw = err_rw./(1-err_rw);
err_kr = err_kr./(1-err_kr);
err_oja = err_oja./(1-err_oja);
err_de = err_de./(1-err_de);

%%
% sample complexity
figure; hold on; box on;
color = get(gca,'colorOrder');
plot([1:max_iter],err_rw,'color',color(1,:));
plot([1:max_iter],err_kr,'color',color(2,:));
plot([1:max_iter],err_oja,'color',color(3,:));
plot([1:max_iter],err_de,'color',color(4,:));
set(gca,'yscale','log');
xlabel('iteration number'); ylabel('estimation error');
legend({'random walk','Krasulina','Oja','DeEPCA'});

% communication complexity
figure; hold on; box on;
color = get(gca,'colorOrder');
plot([0:max_iter-1]*2,err_rw,'color',color(1,:));
plot([0:max_iter-1],err_kr,'color',color(2,:));
plot([0:max_iter-1],err_oja,'color',color(3,:));
plot([0:round(max_iter/N)]*(sum(W(:)>0)-N)*K_de,err_de(1:round(max_iter/N)+1),'color',color(4,:));
set(gca,'yscale','log');
xlabel('communication complexity'); ylabel('estimation error');
legend({'random walk','Krasulina','Oja','DeEPCA'});
