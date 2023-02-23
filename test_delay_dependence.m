clear all; clc; close all;
M = 50; % ambient dim
B = 1; % Batch size
K = 1; % K-PCA
N = 25; % number of agents
W = NetworkGen_Ring(N);
% load topology.mat W %mixing matrix
% %[W,neighbor,neigh_num] = NetworkGen_top(N);
% P = W; P(1:N+1:end)=0; P(P>0) = 1; P = P./(sum(P,2)*ones(1,N));
% neighbor = cell(N,1);
% for n = 1 : N
%     % neighbors of the n-th agent
%     neighbor{n} = find(P(:,n)>0);
%     neighbor{n}(neighbor{n}==n) = [];
% end

%load data_batch_alpha10.mat
U_true = orth(randn(M,1));% u^star
u0 = orth(randn(M,1)); %initilization
sigma_n = 0.1;
alpha = 0;

%algorithmic parameters
max_iter = 1E4;%100000;
G = @(x,u)(x*(x'*u) - u*(u'*x*x'*u));%@(x,u)(x*(x'*u));%
decay_flag = 0;
mu = 0.002; %0.0002;%2E-4;
mu_rw = 0.0008; %0.0008;%0.006;%3E-4;%2E-4;
mu_rw2 = 10*mu;
mu_css = 1/B;
mu_c = 1;%1; 
mu_cm = mu; %step-size
mu_s = 1/B;
indx = 1; indx_rw = 1; indx_rw2 = 1;
s = svd(W); s = s(2);
K_de = 5; eta = (1-sqrt(1-s^2))/(1+sqrt(1-s^2));

% generate data
s_temp = randn(K,B);
n_temp = randn(M,B);
X_all = zeros(M,N*B);
for n = 1 : N
    s = alpha*s_temp + randn(K,B)*sqrt(1-alpha^2);
    noise = alpha*n_temp + randn(M,B)*sqrt(1-alpha^2);
    X_m(:,:,n) = U_true*s + noise*sigma_n;
    X_all(:,(n-1)*B+1:n*B) = squeeze(X_m(:,:,n));
end
[U,~,~] = svd(X_all);
U_star = U(:,1);
Cov = X_all * X_all'/N/B;
clear X_all;

%initialization for cyclic walk
u_walk = u0;  u_m = zeros(M,N); err = zeros(max_iter,1);
grad = zeros(M,1); X = X_m; 

for iter = 1 : max_iter
    
    % cyclic walk PCA
    err(iter) = 1 - norm(u_walk'*U_star)^2/norm(u_walk)^2;
    grad = grad + 1/N*(G(X_m(:,:,indx),u_walk) - G(X(:,:,indx),u_m(:,indx)));
    u_m(:,indx) = u_walk;
    if decay_flag == 1
        u_walk = u_walk + mu/ceil(iter/N)*grad;
    else
        u_walk = u_walk + mu*grad;
    end
    %X(:,:,indx) = X_m(:,:,indx);
    indx = mod(indx, N) + 1;  
end
err = err./(1-err);

%%
% sample complexity
figure; hold on; box on;
color = get(gca,'colorOrder');
plot([1:max_iter],err,'color',color(2,:));
set(gca,'yscale','log');
xlabel('iteration number'); ylabel('estimation error');
legend({'cyclic'});

% % communication complexity
% figure; hold on; box on;
% color = get(gca,'colorOrder');
% plot([0:max_iter-1]*2,err,'color',color(2,:));
% set(gca,'yscale','log');
% xlabel('communication complexity'); ylabel('estimation error');
% legend({'cyclic'});
