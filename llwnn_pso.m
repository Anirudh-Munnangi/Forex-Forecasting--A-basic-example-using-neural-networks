clear all
close all
clc
tic
%% importing data
A=importdata('ind.xls');
%% normalizing data
An=(A.Sheet1 - (min(A.Sheet1)))/(max(A.Sheet1)-min(A.Sheet1));
temp=An((length(An)):-1:1);
An=temp;
data=An(121:1216);
%% rearranging data
t=length(data)-7;
for i=1:t
    table(i,1:6)=data(i:(i+5));
    table(i,7)=((sum(table(i,1:6)))/6);
    table(i,8)=data(i+6);
end
% change from here !
%% training the neural network
wls=7;
N=200; % no of outer loops
p=20; % no of particles
% positin
wpos=rand(wls,8,p);
cpos=rand(1,wls,p);
sipos=rand(1,wls,p);
apos=rand(1,wls,p);
bpos=rand(1,wls,p);
% pbest
wpbest=rand(wls,8,p);
cpbest=rand(1,wls,p);
sipbest=rand(1,wls,p);
apbest=rand(1,wls,p);
bpbest=rand(1,wls,p);
% gbest
wgbest=rand(wls,8,1);
cgbest=rand(1,wls,1);
sigbest=rand(1,wls,1);
agbest=rand(1,wls,1);
bgbest=rand(1,wls,1);
% velocity
velw=rand(wls,8,p);
velc=rand(1,wls,p);
velsi=rand(1,wls,p);
vela=rand(1,wls,p);
velb=rand(1,wls,p);
wmax=1;
wmin=0.3;
wei=linspace(wmax,wmin,N);
c1=1.05; % between 1 and 4 or 1.4962 . . was 1.05, good at 1.55
c2=1.05; % between 1 and 4 or 1.4962  . . was 1.05 good at 1.55
% wei=0.7; % between 0.3 and 1 or 0.7986 . . was 0.5
%% calculating initial fitness
for i=1:p
    fit_valse(i,1:2)=my_fitness_nor(apos(:,:,i),bpos(:,:,i),wpos(:,:,i),cpos(:,:,i),sipos(:,:,i),t,wls,table);
end
    fit_valtms=fit_valse(:,1);
    fit_valtma=fit_valse(:,2);
    fit_y=[];
    err_y=[];
    minfit=10000;
%% pso training loop
for outer=1:N
   
    for k=1:p
       % calculating fitness of the trial vector generated
        [fval(k,:) yorig(k,:) errorig(k,:)]=my_fitness_nor(apos(:,:,k),bpos(:,:,k),wpos(:,:,k),cpos(:,:,k),sipos(:,:,k),t,wls,table);
    end
    % calculating pbest of each particle
    for i=1:p
        if(fit_valtms(i) >= fval(i,1))
            fit_valtms(i)=fval(i,1);
            fit_valtma(i)=fval(i,2);
            fit_y(i,:)=yorig(i,:);
            err_y(i,:)=errorig(i,:);
            wpbest(:,:,i)=wpos(:,:,i);
            cpbest(:,:,i)=cpos(:,:,i);
            sipbest(:,:,i)=sipos(:,:,i);
            apbest(:,:,i)=apos(:,:,i);
            bpbest(:,:,i)=bpos(:,:,i);
        end
    end
    % calulating min fitness and allocating gbest
     for i=1:p
         if(minfit>=fit_valtms(i))
             minfit=fit_valtms(i);
             loc_best=i;
         end
     end
     wgbest =wpbest(:,:,loc_best);
     cgbest =cpbest(:,:,loc_best);
     sigbest=sipbest(:,:,loc_best);
     agbest=apbest(:,:,loc_best);
     bgbest=bpbest(:,:,loc_best);
     fitmse(outer,:)=[minfit fit_valtma(loc_best)];
     yout=fit_y(loc_best,:);
     errout=err_y(loc_best,:);
    % varying the parameters
    for i=1:p
        velw(:,:,i)=(wei(outer)).*velw(:,:,i) + c1*rand(1).*((wpbest(:,:,i))-wpos(:,:,i))+(c2*rand(1).*(wgbest-wpos(:,:,i)));
        velc(:,:,i)=(wei(outer)).*velc(:,:,i) + c1*rand(1).*((cpbest(:,:,i))-cpos(:,:,i))+(c2*rand(1).*(cgbest-cpos(:,:,i)));
        velsi(:,:,i)=(wei(outer)).*velsi(:,:,i) + c1*rand(1).*((sipbest(:,:,i))-sipos(:,:,i))+(c2*rand(1).*(sigbest-sipos(:,:,i)));
        vela(:,:,i)=(wei(outer)).*vela(:,:,i) + c1*rand(1).*((apbest(:,:,i))-apos(:,:,i))+(c2*rand(1).*(agbest-apos(:,:,i)));
        velb(:,:,i)=(wei(outer)).*velb(:,:,i) + c1*rand(1).*((bpbest(:,:,i))-bpos(:,:,i))+(c2*rand(1).*(bgbest-bpos(:,:,i)));
    end
    for i=1:p
        % no abs earlier
        wpos(:,:,i)=wpos(:,:,i)+velw(:,:,i);
        cpos(:,:,i)=cpos(:,:,i)+velc(:,:,i);
        sipos(:,:,i)=sipos(:,:,i)+velsi(:,:,i);
        apos(:,:,i)=apos(:,:,i)+vela(:,:,i);
        bpos(:,:,i)=bpos(:,:,i)+velb(:,:,i);
    end
end

figure
plot(A.Sheet1)
title(' original data used for training');
xlabel('sample number');
ylabel('magnitude');

figure
hold on
plot(yout,'b');
plot(table(1:t,7), 'k');
title( ' network output versus original output during training ' );
xlabel('no of samples');
ylabel('magnitude');
legend('network output best particle','original output');

figure
plot(errout);
title(' error between original and predicted during training ');

figure
plot(fitmse(1:outer-1,1));
title('rmse plot wrt iterations during trainng');

figure
plot(fitmse(1:outer-1,2));
toc
             
