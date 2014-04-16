% changed the value of dsq's by (x-b)/a
% updated the values of ai and bi similar to bias
% derivatives old as in paper
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
%% training the neural network
f=0.8;
co=0.5;
wls=7;
rate=0.72;
np=20;
out=500;
aorig=rand(1,wls,np);
borig=rand(1,wls,np);
a1=rand(1,wls,np);
b1=rand(1,wls,np);
a2=rand(1,wls,np);
b2=rand(1,wls,np);
worig=rand(wls,8,np);
corig=rand(1,wls,np);
siorig=rand(1,wls,np);
w1=rand(wls,8,np);
c1=rand(1,wls,np);
si1=rand(1,wls,np);
w2=rand(wls,8,np);
c2=rand(1,wls,np);
si2=rand(1,wls,np);
cfzh=rand(1,7,np);
sifzh=rand(1,7,np);
yorigt=[];
erroirgt=[];
for i=1:np
cfzl(1,1:7,i)=(ones(1,7)-cfzh(1,1:7,i))/100;
sifzl(1,1:7,i)=(ones(1,7)-sifzh(1,1:7,i))/100;
end
% cfz=rand(1,14);
% sifz=rand(1,14);

%% LLWNN only
% calculating initial fitnesses
for outer=1:np
    fit_val(outer,:)=my_fitness_nor(aorig(:,:,outer),borig(:,:,outer),worig(:,:,outer),corig(:,:,outer),siorig(:,:,outer),t,wls,table);
end
%% De training loops
    for outer=1:out
        fit_valt=fit_val;
     for inner=1:np
         ctr=0;
         while(ctr==0)
          choice=ceil(10*rand(1,3));
          if(choice(1)~=choice(2) && choice(2)~=choice(3) && choice(1)~=choice(3))
              ctr=1;
          end
         end
        % target set two sets as your must know
        % one for llwnn and another for neurofuzzyllwnn
        aorigt=aorig(:,:,inner);
        borigt=borig(:,:,inner);
        worigt=worig(:,:,inner);
        corigt=corig(:,:,inner);
        siorigt=siorig(:,:,inner);
       
       
        
        % sets for the randomly chosen vectors
        % set 1
        aorig1=aorig(:,:,choice(1));
        borig1=borig(:,:,choice(1));
        worig1=worig(:,:,choice(1));
        corig1=corig(:,:,choice(1));
        siorig1=siorig(:,:,choice(1));
        
        % set 2
        aorig2=aorig(:,:,choice(2));
        borig2=borig(:,:,choice(2));
        worig2=worig(:,:,choice(2));
        corig2=corig(:,:,choice(2));
        siorig2=siorig(:,:,choice(2));
        
        
        % set 3
        aorig3=aorig(:,:,choice(3));
        borig3=borig(:,:,choice(3));
        worig3=worig(:,:,choice(3));
        corig3=corig(:,:,choice(3));
        siorig3=siorig(:,:,choice(3));
        
        
        % calculating difference and multiplying the factor f
        aorigd=(aorig1-aorig2)*f;
        borigd=(borig1-borig2)*f;
        worigd=(worig1-worig2)*f;
        corigd=(corig1-corig2)*f;
        siorigd=(siorig1-siorig2)*f;
       
        
        % generating the noisy random vector by summing the difference and
        % the other randomly generated set.
        aorignr=aorig3+aorigd;
        borignr=borig3+borigd;
        worignr=worig3+worigd;
        corignr=corig3+corigd;
        siorignr=siorig3+siorigd;
        
        
        % crossover operation and generating trial vector
        rn=rand(1,5);
        if(rn(1)>0.5)
            aorigtv=aorigt;
        else
            aorigtv=aorignr;
        end
        
        if(rn(2)>0.5)
            borigtv=borigt;
        else
            borigtv=borignr;
        end
        
       if(rn(3)>0.5)
            worigtv=worigt;
        else
            worigtv=worignr;
        end
        
        if(rn(4)>0.5)
            corigtv=corigt;
        else
            corigtv=corignr;
        end
        
        if(rn(5)>0.5)
            siorigtv=siorigt;
        else
            siorigtv=siorignr;
        end
        
       
        % calculating fitness of the trial vector generated
        [fval(inner,:) yorig errorig]=my_fitness_nor(aorigtv,borigtv,worigtv,corigtv,siorigtv,t,wls,table);
        if(fval(inner,1)<fit_valt(inner,1))
            aorignew(:,:,inner)=aorigtv;
            borignew(:,:,inner)=borigtv;
            worignew(:,:,inner)=worigtv;
            corignew(:,:,inner)=corigtv;
            siorignew(:,:,inner)=siorigtv;
        else
            aorignew(:,:,inner)=aorigt;
            borignew(:,:,inner)=borigt;
            worignew(:,:,inner)=worigt;
            corignew(:,:,inner)=corigt;
            siorignew(:,:,inner)=siorigt;
            fval(inner,:)=fit_valt(inner,:);
        end
        % initialize the new set to original
        % store the fitness value. . how check
     end
            aorig=aorignew;
            borig=borignew;
            worig=worignew;
            corig=corignew;
            siorig=siorignew;
            fit_val=fval;
        fitmse(outer)=fval(1,1);
        fitmap(outer)=fval(1,2);
    % new trial set . .there are two set as you must remember
    % one for llwnn and the other for neurofuzzy-llwnn
end

%% plotting results of training
figure
plot(A.Sheet1)
title(' Exchange rate : Dollar in INR www.oanda.com');
ylabel('Original data value');
xlabel('No of input patterns');

figure
hold on
plot(yorig,'k');
% plot(Ynet,'b');
plot(table(1:t,8), 'r');
title( ' Comparision of estimated values wrt Normalized data set' );
% legend('llwnn output','llwnn+neurofuzzy output','original output');
legend('llwnn output','original output');
ylabel('Normalized data value');
xlabel('No of imput patterns');
hold off

 figure
 hold on
 plot(errorig,'b');
%  plot(errtot,'r');
%  legend('llwnn error', 'llwnn + neurofuzzy error');
 legend('llwnn error');
 title(' error  ');
 ylabel('magnitude');
 xlabel('No of input patterns');
 hold off
 
 figure
 hold on
%  plot(mseporig,'b');
plot(fitmse,'b');
%  plot(mseptot,'r');
%  legend('llwnn rmse','llwnn+neurofuzzy rmse');
 legend('llwnn rmse');
 title('rmse plot wrt iterations during trainng');
 ylabel('magnitude');
 xlabel('No of iterations');
 hold off
 
 figure
 hold on
%  plot(mapeporig,'b');
%  plot(mapeptot,'r');
 plot(fitmap,'b');
 title(' mape plot wrt iterations during training');
%  legend('llwnn mape', 'llwnn+neurofuzzy mape');
legend('llwnn mape');
 ylabel('magnitude');
 xlabel('No of iterations');
 hold off
toc
