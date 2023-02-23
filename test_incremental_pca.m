clear all
rnd = randi(1e5);
rng(100) %

% problem setting
k = 1; % K-PCA
eps = 1e-10;
numepochs = 20;

% % generate synthetic data
% d = 1000; % ambient dim
% n = 50000; % sample size
% s = randn(k,n);
% U_true = orth(randn(d,k));% u^star
% noise = randn(d,n);
% sigma_n = 1;
% X_m = U_true*s + noise*sigma_n;
% [U,~,~] = svds(sparse(X_m));
% U_star = U(:,1);
% Cov = X_m * X_m'/n;

% load libsvm data
[b, A] = libsvmread('data\a9a');
[n, d] = size(A);
X_m = full(A)';
[U,~,~] = svd(X_m);
U_star = U(:,1);
Cov = X_m * X_m'/n;

% load cifar10.mat
% [n, d] = size(Z);
% X_m = Z';
% [U,~,~] = svds(sparse(X_m));
% U_star = U(:,1);
% Cov = X_m * X_m'/n;

% initilization
u0 = orth(randn(d,1)); 

%% Krasulina
disp('Krasulina');
eta_Krasulina = 1e-4; % 0.002;
[err_Krasulina] = Krasulina(X_m,numepochs,u0,U_star,eta_Krasulina,eps);

%% Oja
disp('Oja');
eta_Oja = 1e-4; % 0.001;
[err_Oja] = Oja(X_m,numepochs,u0,U_star,eta_Oja,eps);

%% VR-PCA
disp('VR-PCA');
lambda1 = eigs(Cov,1);
[err_VRPCA] = VRPCA(X_m,numepochs,u0,U_star,eps);

%% VR Power+M
disp('VR Power+M');
s = eigs(Cov,2);
lambda2 = s(2);
err_VRPowerM = VRPowerM(X_m,numepochs,u0,U_star,lambda2,eps);

%% VR Power
disp('VR Power');
eta_VRPower = 1e-4; % 1e-3 for sigma = 0.5 (synthetic), 1e-4 for sigma = 1 (synthetic)
[err_VRPower] = VRPower(X_m,numepochs,u0,U_star,eta_VRPower,eps); 

%% IARG-PCA
disp('IARG-PCA');
eta_IARG = 2e-5; % 2e-5 for sigma = 0.5 (synthetic)
[err_IARG] = IARG(X_m,numepochs,u0,U_star,eta_IARG,eps);

%% plot
figure;
xran = 0:2:numepochs-2;
p = semilogy(xran,err_Krasulina,'-m','LineWidth',2);
p.LineStyle = '--';
hold on;
p = semilogy(xran,err_Oja,'Color','#77AC30','LineWidth',2);
p.LineStyle = '-.';
hold on;
p = semilogy(xran,err_VRPCA,'-b','LineWidth',2);
p.LineStyle = ':';
hold on;
p = semilogy(xran,err_VRPowerM,'-k','LineWidth',2);
p.LineStyle = '-.';
hold on;
p = semilogy(xran,err_VRPower,'Color','#A2142F','LineWidth',2);
p.LineStyle = '--';
hold on;
p = semilogy(xran,err_IARG,'-r','LineWidth',2);
p.LineStyle = '-';

ax = gca;
ax.FontSize = 15;
xlabel('Number of Passed Data Samples ($\times 10^4$)','Interpreter','latex','FontSize',20);
ylabel('{$1-\langle \textbf{{\emph w}}^t,\textbf{{\emph v}}_1 \rangle^2$}','Interpreter','latex','FontSize',25);
lgd = legend('Krasulina','Oja','VR-PCA','VR Power+M','VR Power','IARG-PCA','location','southwest');
lgd.FontSize = 13;
% xlim([1,10]);

