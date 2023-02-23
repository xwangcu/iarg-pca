
currentFolder = pwd;
load([currentFoder,'/cifar/Results/result_cifar_ring.mat']); 
% load([currentFoder,'/cifar/Results/result_cifar_er.mat']); 

figure; 
max_iter = length(err_rw);
N = size(W,1);
% subplot(1,4,k);
% hold on; box on;
% color = get(gca,'colorOrder');
% plot((1:max_iter),err_rw,'color',color(1,:),'LineWidth',1);
% plot((1:max_iter),err_kr,'color',color(2,:),'LineWidth',1);
% plot((1:max_iter),err_oja,'color',color(3,:),'LineWidth',1);
% plot((1:max_iter),err_de,'color',color(4,:),'LineWidth',1);
% set(gca,'yscale','log');
% xlabel('iteration number'); 
% if k==1
%     ylabel('estimation error');
% end
% title(tit);
% xlim([0,max_iter]);

% communication complexity
% subplot(1,4,k+1);
hold on; 
box on;
color = get(gca,'colorOrder');
plot((0:max_iter-1),err_kr,'color','b','LineWidth',2.5);
plot((0:max_iter-1),err_oja,'color','#EDB120','LineWidth',1.5,'LineStyle', '-.');
plot((0:round(max_iter/N))*(sum(W(:)>0)-N)*K_de,err_de(1:round(max_iter/N)+1),'color','k','LineWidth',1.5,'LineStyle', '--');
plot((0:max_iter-1)*2,err_rw,'color','r','LineWidth',1.5);

% set(gca,'yscale','log');
% xlabel('Communication Cost');
% ylabel('{$1-\langle \textbf{{\emph w}}^t,\textbf{{\emph v}}_1 \rangle^2$}','Interpreter','latex','FontSize',25);
% if k==1
%     legend({'IARG-PCA','Krasulina','Oja','DeEPCA'});
% end
% title(tit);
% xlim([1,1e4]);

set(gca,'yscale','log');
ax = gca;
ax.FontSize = 15;
ax.XAxis.Exponent = 3;
xlabel('Communication Cost','FontSize',20);
ylabel('{$1-\langle \textbf{{\emph w}}^t,\textbf{{\emph v}}_1 \rangle^2$}','Interpreter','latex','FontSize',25);
lgd = legend('Krasulina','Oja','DeEPCA','IARG-PCA','location','southeast');
lgd.FontSize = 13;
title('Ring (CIFAR-10)');
xlim([1,1e4]);
% ylim([0,1e-10]);
