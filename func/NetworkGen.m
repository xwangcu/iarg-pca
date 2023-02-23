function [cmat,incimat,nnum,coordinates]=NetworkGen(J,xmax,ymax,CR)

while 1
    % --- generating agents ---
    xpos=xmax*rand(J,1);            % J processors
    ypos=ymax*rand(J,1);
    coordinates = [xpos, ypos];
    
    % --- connectivity matrix ---
    dmat=(repmat(xpos,1,J)-repmat(xpos',J,1)).^2+(repmat(ypos,1,J)-repmat(ypos',J,1)).^2;
    cmat=(dmat<=CR^2);
    cmat=cmat.*(1-eye(J));
    r = zeros(1, J);
    r(2) = 1;
    r(J) = 1;
    ring = toeplitz(r);
    %cmat= cmat+ ring;
    cmat = double(cmat>0);
    % --- number of neighbors ---
    nnum=sum(cmat,2);
    % --- check connectivity ---
    nodeclass.conmatrix=cmat;
    flag1con=verify1con(nodeclass);
    if flag1con==1
        disp('Network Connected!');
        break;
    end
end
incimat=size(J,sum(sum(cmat))/2);
count=1;
for i=1:J-1
    for j=i+1:J
        if cmat(i,j)==1
            incimat(i,count)=1;
            incimat(j,count)=-1;
            count=count+1;
            incimat(j,count)=1;
            incimat(i,count)=-1;
            count=count+1;
        end
    end
end