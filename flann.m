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
w=rand(1,18);
f=0.24;
N=100;
for i=1:N
    for k=1:t
        x(1:6)=table(k,1:6);
        cospart(1:6)=cos(pi.*x(1:6));
        sinpart(1:6)=sin(pi.*x(1:6));
        topsigc=w(1:2:11)*(cospart');
        topsigs=w(2:2:12)*(sinpart');
        y1=topsigc+topsigs;
        botsig=w(13:18)*(x');
        ynet=y1+botsig;
        op(k)=tanh(ynet);
        err(k)=table(k,7)-op(k);
        errsq(k)=((err(k))^2);
        mape(k)=abs((err(k)/table(k,7)));
        % training the weights
        w(13:18)=w(13:18) +f*(table(k,1:6))*((sech(ynet))^2)*err(k);
        w(1:2:11)=w(1:2:11) +f*cos(pi.*(table(k,1:6)))*((sech(ynet))^2)*err(k);
        w(2:2:12)=w(2:2:12) +f*sin(pi.*(table(k,1:6)))*((sech(ynet))^2)*err(k);
    end
    msep(i)=((sum(errsq))/t);
    mapep(i)=((sum(mape))/t)*100;
end
%% plotting results of training
figure
plot(A.Sheet1)
title(' original data used for training');
figure
hold on
plot(op,'b');
plot(table(1:t,7), 'r');
title( ' network output versus original output during training ' );
legend(' network output','original output');
xlabel('sample number ');
ylabel('magnitude');

figure
plot(err);
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
