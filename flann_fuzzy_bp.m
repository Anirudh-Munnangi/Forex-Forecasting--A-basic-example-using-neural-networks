clear all
clc
close all
%% importing data
A=importdata('ind_source_full.xls');
%% normalizing data
An=(A.Sheet1 - (min(A.Sheet1)))/(max(A.Sheet1)-min(A.Sheet1));
data=An(300:1800);
%% rearranging data
t=length(data)-7;
for i=1:t
    table(i,1:7)=data(i:(i+6));
end
%% training the neural network
w1=rand(1,18);
w2=rand(1,18);
f=0.24;
N=100;
cfzh=rand(1,6);
sifzh=rand(1,6);
cfzl(1:6)=(ones(1,6)-cfzh)/100;
sifzl(1:6)=(ones(1,6)-sifzh)/100;
for i=1:N
    for k=1:t
        % part 1 flann
        x(1:6)=table(k,1:6);
        cospart1(1:6)=cos(pi.*x(1:6));
        sinpart1(1:6)=sin(pi.*x(1:6));
        topsigc1=w1(1:2:11)*(cospart1');
        topsigs1=w1(2:2:12)*(sinpart1');
        y1=topsigc1+topsigs1;
        botsig1=w1(13:18)*(x');
        ynet1=y1+botsig1;
        op1(k)=tanh(ynet1);
        
        % part 2 flann
        x(1:6)=table(k,1:6);
        cospart2(1:6)=cos(pi.*x(1:6));
        sinpart2(1:6)=sin(pi.*x(1:6));
        topsigc2=w2(1:2:11)*(cospart2');
        topsigs2=w2(2:2:12)*(sinpart2');
        y2=topsigc2+topsigs2;
        botsig2=w2(13:18)*(x');
        ynet2=y2+botsig2;
        op2(k)=tanh(ynet2);
        
          % FUZZY part
      
      x(1:6)=table(k,1:6);
      x1f1(k)=exp((-1)*((x(1)-cfzh(1))^2)/((sifzh(1))^2));
      x1f2(k)=exp((-1)*((x(1)-cfzl(1))^2)/((sifzl(1))^2));
      x2f1(k)=exp((-1)*((x(2)-cfzh(2))^2)/((sifzh(2))^2));
      x2f2(k)=exp((-1)*((x(2)-cfzl(2))^2)/((sifzl(2))^2));
      x3f1(k)=exp((-1)*((x(3)-cfzh(3))^2)/((sifzh(3))^2));
      x3f2(k)=exp((-1)*((x(3)-cfzl(3))^2)/((sifzl(3))^2));
      x4f1(k)=exp((-1)*((x(4)-cfzh(4))^2)/((sifzh(4))^2));
      x4f2(k)=exp((-1)*((x(4)-cfzl(4))^2)/((sifzl(4))^2));
      x5f1(k)=exp((-1)*((x(5)-cfzh(5))^2)/((sifzh(5))^2));
      x5f2(k)=exp((-1)*((x(5)-cfzl(5))^2)/((sifzl(5))^2));
      x6f1(k)=exp((-1)*((x(6)-cfzh(6))^2)/((sifzh(6))^2));
      x6f2(k)=exp((-1)*((x(6)-cfzl(6))^2)/((sifzl(6))^2));
      
      pf1(k)=min([x1f1(k) x2f1(k) x3f1(k) x4f1(k) x5f1(k) x6f1(k)]);
      pf2(k)=min([x1f2(k) x2f2(k) x3f2(k) x4f2(k) x5f2(k) x6f2(k)]);
      
      Ynet1(k)=pf1(k)*op1(k);
      Ynet2(k)=pf2(k)*op2(k);
      Ynet(k)=Ynet1(k)+Ynet2(k);
      Ynet(k)=Ynet(k)/(pf1(k)+pf2(k));
      % calculating errors
     
        err1(k)=table(k,7)-Ynet(k);
        errsq1(k)=((err1(k))^2);
        mape1(k)=abs((err1(k)/table(k,7)));
        err2(k)=table(k,7)-Ynet(k);
        errsq2(k)=((err2(k))^2);
        mape2(k)=abs((err2(k)/table(k,7)));

        % training the weights of both part1 and part2 flann
        % part 1 flann
        w1(13:18)=w1(13:18) +f*(table(k,1:6))*((sech(Ynet(k)))^2)*err1(k);
        w1(1:2:11)=w1(1:2:11) +f*cos(pi.*(table(k,1:6)))*((sech(Ynet(k))^2))*err1(k);
        w1(2:2:12)=w1(2:2:12) +f*sin(pi.*(table(k,1:6)))*((sech(Ynet(k)))^2)*err1(k);
        % part 2 flann
        w2(13:18)=w2(13:18) +f*(table(k,1:6))*((sech(Ynet(k)))^2)*err2(k);
        w2(1:2:11)=w2(1:2:11) +f*cos(pi.*(table(k,1:6)))*((sech(Ynet(k))^2))*err2(k);
        w2(2:2:12)=w2(2:2:12) +f*sin(pi.*(table(k,1:6)))*((sech(Ynet(k)))^2)*err2(k);
    end
    msep(i)=((sum((errsq1+errsq2)/2))/t);
    mapep(i)=((sum((mape1+mape2)/2))/t)*100;
end
%% plotting results of training
figure
plot(A.Sheet1)
title(' original data used for training');
figure
hold on
plot(Ynet,'b');
plot(table(1:t,7), 'r');
title( ' network output versus original output during training ' );
legend(' network output','original output');
xlabel('sample number ');
ylabel('magnitude');

figure
plot((err1+err2)/2);
title(' error between original and predicted during training ');
xlabel('sample number ');
ylabel('magnitude');

figure
plot(msep);
title('rmse plot wrt iterations during trainng');
xlabel('sample number ');
ylabel('magnitude');

figure
plot(mapep);
title(' mape plot wrt iterations during training');
xlabel('sample number ');
ylabel('magnitude');
