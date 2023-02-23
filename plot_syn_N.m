load ../Results/result_N.mat

figure;hold on;box on;
color = get(gca,'colorOrder');
c1 = sum(N_array.*compl_rw)/norm(N_array)^2;
c2 = sum(N_array.^2.*compl_de)/norm(N_array.^2)^2;
plot(N_array,compl_rw,'.','Marker','o','MarkerSize',6,'MarkerFaceColor',color(1,:));
plot(N_array,c1*N_array,'-','LineWidth',1,'color',color(1,:));
plot(N_array,compl_de/10,'.','Marker','o','MarkerSize',6,...
    'MarkerFaceColor',color(4,:),'MarkerEdgeColor',color(4,:));
plot(N_array,c2*N_array.^2/10,'-','LineWidth',1,'color',color(4,:));
ylabel('communication complexity','interpreter','latex');
xlabel('$m$','interpreter','latex');
legend({'IARG-PCA','$O(N)$',...
    'DeEPCA($\times 0.1$)','$O(N^2)$'},...
    'interpreter','latex');
ylim([0,6E3]);