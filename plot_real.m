load ../Results/result_w8a_ring_all.mat
figure;
myplot(err_rw,err_kr,err_oja,err_de,W,K_de,'ring graph',1);
load ../Results/result_w8a_er_all.mat
myplot(err_rw,err_kr,err_oja,err_de,W,K_de,'ER graph',3);

clear all;
load ../Results/result_a9a_ring_all.mat
figure;
myplot(err_rw,err_kr,err_oja,err_de,W,K_de,'ring graph',1);
load ../Results/result_a9a_er_all.mat
myplot(err_rw,err_kr,err_oja,err_de,W,K_de,'ER graph',3);

function myplot(err_rw,err_kr,err_oja,err_de,W,K_de,tit,k)
%figure; 
max_iter = length(err_rw);
N = size(W,1);
subplot(1,4,k);
hold on; box on;
color = get(gca,'colorOrder');
plot([1:max_iter],err_rw,'color',color(1,:),'LineWidth',1);
plot([1:max_iter],err_kr,'color',color(2,:),'LineWidth',1);
plot([1:max_iter],err_oja,'color',color(3,:),'LineWidth',1);
plot([1:max_iter],err_de,'color',color(4,:),'LineWidth',1);
set(gca,'yscale','log');
xlabel('iteration number'); 
if k==1
    ylabel('estimation error');
end
title(tit);
xlim([0,max_iter]);

% communication complexity
subplot(1,4,k+1);
hold on; box on;
color = get(gca,'colorOrder');
plot([0:max_iter-1]*2,err_rw,'color',color(1,:),'LineWidth',1);
plot([0:max_iter-1],err_kr,'color',color(2,:),'LineWidth',1);
plot([0:max_iter-1],err_oja,'color',color(3,:),'LineWidth',1);
plot([0:round(max_iter/N)]*(sum(W(:)>0)-N)*K_de,err_de(1:round(max_iter/N)+1),'color',color(4,:),'LineWidth',1);
set(gca,'yscale','log');
xlabel('communication complexity'); 
%ylabel('estimation error');
if k==1
    legend({'IARG-PCA','Krasulina','Oja','DeEPCA'});
end
%legend({'random walk','Krasulina','Oja','DeEPCA'});
title(tit);
xlim([1,1E4]);
end