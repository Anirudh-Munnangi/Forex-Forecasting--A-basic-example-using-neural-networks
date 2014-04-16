clear all
close all
clc
%% importing data
A=importdata('ind_source_full.xls');
%% normalizing data
An=(A.Sheet1 - (min(A.Sheet1)))/(max(A.Sheet1)-min(A.Sheet1));
data=An(100:1800);
%% rearranging data
t=length(data)-7;
for i=1:t
    table(i,1:7)=data(i:(i+6));
end
%% training the neural network
wls=10;
outerlen=50;
rate=0.5;
worig=rand(wls,7);
corig=rand(1,wls);
siorig=rand(1,wls);
w1=rand(wls,7);
c1=rand(1,wls);
si1=rand(1,wls);
w2=rand(wls,7);
c2=rand(1,wls);
si2=rand(1,wls);
cfzh=rand(1,6);
sifzh=rand(1,6);
cfzl(1:6)=(ones(1,6)-cfzh)/100;
sifzl(1:6)=(ones(1,6)-sifzh)/100;
for out=1:outerlen
    for in=1:t
        %calculating sigma products before applying basis function 
        
        for i=1:wls
       sigma(i)=worig(i,1)+(worig(i,2:7))*((table(in,1:6))');
       end
       d=sum((table(in,1:6)).^2);
       dsq=sqrt(d);
      %calculation of the psi's
      for i=1:wls
          inside=(dsq-corig(i))/(siorig(i));
          insq=(inside)^2;
          psi(i)=exp(-1*insq);
      end
      % calculation of the final output
      for i=1:wls
          prod(i)=sigma(i)*psi(i);
      end
      yorig(in)=sum(prod);
      
       % calculation of error of only llwnn
      errorig(in)=table(in,7)-yorig(in);
      errsqorig(in)=(errorig(in))^2;
      mapeorig(in)=abs((errorig(in))/(table(in,7)));
      
      % updating biases
    % short cut wala w(1:10,1)=w(1:10,1)+rate*err(in)*psi(1:10);
    for i=1:wls
        worig(i,1)=worig(i,1)+rate*errorig(in)*psi(i);
    end
    % updating weights
    for i=1:wls
        for j=2:7
            worig(i,j)=worig(i,j)+rate*errorig(in)*psi(i)*table(in,(j-1));
        end
    end
    % updating c
    for i=1:wls
        sig=worig(i,1)+ (worig(i,2:7))*((table(in,1:6))');
        del=2*(((dsq-corig(i))^2)/((siorig(i))^3))*(exp((-1*((dsq-corig(i))^2))/((siorig(i))^2)));
        pro=rate*(errorig(in))*sig*del;
        siorig(i)=siorig(i)+pro;
    end
    
    % updating si
    for i=1:wls
        sig=worig(i,1)+ (worig(i,2:7))*((table(in,1:6))');
        del=4*((corig(i))/((siorig(i))^2))*(exp((-1*((dsq-corig(i))^2))/((siorig(i))^2)));
        pro=rate*(errorig(in))*sig*del;
        corig(i)=corig(i)+pro;
    end
      
   %---------------------------------------------------------------------     
      %calculating sigma products before applying basis function of
      %neurofuzzy part
       for i=1:wls
       sigma(i)=w1(i,1)+(w1(i,2:7))*((table(in,1:6))');
       end
       d=sum((table(in,1:6)).^2);
       dsq=sqrt(d);
      %calculation of the psi's
      for i=1:wls
          inside=(dsq-c1(i))/(si1(i));
          insq=(inside)^2;
          psi(i)=exp(-1*insq);
      end
      % calculation of the final output
      for i=1:wls
          prod(i)=sigma(i)*psi(i);
      end
      y1(in)=sum(prod);
       
      % calculating sigma products before applying basis function
       for i=1:wls
       sigma(i)=w2(i,1)+(w2(i,2:7))*((table(in,1:6))');
       end
       d=sum((table(in,1:6)).^2);
       dsq=sqrt(d);
      %calculation of the psi's
      for i=1:wls
          inside=(dsq-c2(i))/(si2(i));
          insq=(inside)^2;
          psi(i)=exp(-1*insq);
      end
      % calculation of the final output
      for i=1:wls
          prod(i)=sigma(i)*psi(i);
      end
      y2(in)=sum(prod);
      
      % FUZZY part
      
      x(1:6)=table(in,1:6);
      x1f1(in)=exp((-1)*((x(1)-cfzh(1))^2)/((sifzh(1))^2));
      x1f2(in)=exp((-1)*((x(1)-cfzl(1))^2)/((sifzl(1))^2));
      x2f1(in)=exp((-1)*((x(2)-cfzh(2))^2)/((sifzh(2))^2));
      x2f2(in)=exp((-1)*((x(2)-cfzl(2))^2)/((sifzl(2))^2));
      x3f1(in)=exp((-1)*((x(3)-cfzh(3))^2)/((sifzh(3))^2));
      x3f2(in)=exp((-1)*((x(3)-cfzl(3))^2)/((sifzl(3))^2));
      x4f1(in)=exp((-1)*((x(4)-cfzh(4))^2)/((sifzh(4))^2));
      x4f2(in)=exp((-1)*((x(4)-cfzl(4))^2)/((sifzl(4))^2));
      x5f1(in)=exp((-1)*((x(5)-cfzh(5))^2)/((sifzh(5))^2));
      x5f2(in)=exp((-1)*((x(5)-cfzl(5))^2)/((sifzl(5))^2));
      x6f1(in)=exp((-1)*((x(6)-cfzh(6))^2)/((sifzh(6))^2));
      x6f2(in)=exp((-1)*((x(6)-cfzl(6))^2)/((sifzl(6))^2));
      
      pf1(in)=min([x1f1(in) x2f1(in) x3f1(in) x4f1(in) x5f1(in) x6f1(in)]);
      pf2(in)=min([x1f2(in) x2f2(in) x3f2(in) x4f2(in) x5f2(in) x6f2(in)]);
      
      Ynet1(in)=pf1(in)*y1(in);
      Ynet2(in)=pf2(in)*y2(in);
      Ynet(in)=Ynet1(in)+Ynet2(in);
      Ynet(in)=Ynet(in)/(pf1(in)+pf2(in));
      
      % calculation of error of first part
      err1(in)=table(in,7)-Ynet(in);
      errsq1(in)=(err1(in))^2;
      mape1(in)=abs((err1(in))/(table(in,7)));
      % updating parameters
    % updating biases
    % short cut wala w(1:10,1)=w(1:10,1)+rate*err(in)*psi(1:10);
    for i=1:wls
        w1(i,1)=w1(i,1)+rate*err1(in)*psi(i);
    end
    % updating weights
    for i=1:wls
        for j=2:7
            w1(i,j)=w1(i,j)+rate*err1(in)*psi(i)*table(in,(j-1));
        end
    end
    % updating c
    for i=1:wls
        sig=w1(i,1)+ (w1(i,2:7))*((table(in,1:6))');
        del=2*(((dsq-c1(i))^2)/((si1(i))^3))*(exp((-1*((dsq-c1(i))^2))/((si1(i))^2)));
        pro=rate*(err1(in))*sig*del;
        si1(i)=si1(i)+pro;
    end
    
    % updating si
    for i=1:wls
        sig=w1(i,1)+ (w1(i,2:7))*((table(in,1:6))');
        del=4*((c1(i))/((si1(i))^2))*(exp((-1*((dsq-c1(i))^2))/((si1(i))^2)));
        pro=rate*(err1(in))*sig*del;
        c1(i)=c1(i)+pro;
    end

      % calculation of error of second part
      err2(in)=table(in,7)-Ynet(in);
      errsq2(in)=(err2(in))^2;
      mape2(in)=abs((err2(in))/(table(in,7)));
      % updating parameters
    % updating biases
    % short cut wala w(1:10,1)=w(1:10,1)+rate*err(in)*psi(1:10);
    for i=1:wls
        w2(i,1)=w2(i,1)+rate*err2(in)*psi(i);
    end
    % updating weights
    for i=1:wls
        for j=2:7
            w2(i,j)=w2(i,j)+rate*err2(in)*psi(i)*table(in,(j-1));
        end
    end
    % updating c
    for i=1:wls
        sig=w2(i,1)+ (w2(i,2:7))*((table(in,1:6))');
        del=2*(((dsq-c2(i))^2)/((si2(i))^3))*(exp((-1*((dsq-c2(i))^2))/((si2(i))^2)));
        pro=rate*(err2(in))*sig*del;
        si2(i)=si2(i)+pro;
    end
    
    % updating si
    for i=1:wls
        sig=w2(i,1)+ (w2(i,2:7))*((table(in,1:6))');
        del=4*((c2(i))/((si2(i))^2))*(exp((-1*((dsq-c2(i))^2))/((si2(i))^2)));
        pro=rate*(err2(in))*sig*del;
        c2(i)=c2(i)+pro;
    end
    end
     msep1(out)=((sum(errsq1))/100);
     mapep1(out)=((sum(mape1))/100);
     msep2(out)=((sum(errsq2))/100);
     mapep2(out)=((sum(mape2))/100);
     mseporig(out)=((sum(errsqorig))/100);
     mapeporig(out)=((sum(mapeorig))/100);
     
end

%% plotting results of training
figure
plot(A.Sheet1)
title(' original data used for training');
ylabel('magnitude');
xlabel('sample number');
figure
hold on
plot(yorig,'k');
plot(Ynet,'b');
plot(table(1:t,7), 'r');
title( ' llwnn output/llwnn+neurofuzzy output/ original training ' );
legend('llwnn output','llwnn+neurofuzzy output','original output');
ylabel('magnitude');
xlabel('sample number');
 figure
 hold on
 plot(errorig,'b');
 plot(((err1+err2)/2),'r');
 legend('llwnn error', 'llwnn + neurofuzzy error');
 title(' error  ');
 ylabel('magnitude');
 xlabel('sample number');
 figure
 hold on
 plot(1:outerlen,(mseporig),'b');
 plot(1:outerlen,((msep1+msep2)/2),'r');
 legend('llwnn rmse','llwnn+neurofuzzy rmse');
 title('rmse plot wrt iterations during trainng');
 ylabel('magnitude');
 xlabel('sample number');
 figure
 hold on
 plot(1:outerlen,mapeporig,'b');
 plot(1:outerlen,((mapep1+mapep2)/2),'r');
 title(' mape plot wrt iterations during training');
 legend('llwnn mape', 'llwnn+neurofuzzy mape');
 ylabel('magnitude');
 xlabel('sample number');
