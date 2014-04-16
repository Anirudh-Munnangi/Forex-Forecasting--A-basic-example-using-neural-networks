clear all
close all
clc
%% importing data
A=importdata('ind_source.xls');
%% normalizing data
An=(A.Sheet1 - (min(A.Sheet1)))/(max(A.Sheet1)-min(A.Sheet1));
data=An(63:429);
%% rearranging data
t=length(data)-7;
for i=1:t
    table(i,1:7)=data(i:(i+6));
end
%% training the neural network
w=rand(1,6);
f=0.24;
for i=1:100
    for k=1:t
        sigma=(table(k,1:6))*(w');
        y(k)=tanh(sigma);
        errort(k)=table(k,7)-y(k);
        errsqt(k)=(errort(k))^2;
        mape(k)=abs((errort(k))/(table(k,7)));
        w(1:6)=w(1:6) +f*(table(k,1:6))*((sech(sigma))^2)*errort(k);
    end
    msep(i)=((sum(errsqt))/100);
    mapep(i)=((sum(mape))/100);
end
%% plotting results of training
figure
plot(A.Sheet1)
title(' original data used for training');
figure
hold on
plot(y,'b');
plot(table(1:t,7), 'r');
title( ' network output versus original output during training ' );
legend(' network output','original output');
figure
plot(errort);
title(' error between original and predicted during training ');
figure
plot(1:100,msep);
title('rmse plot wrt iterations during trainng');
figure
plot(1:100,mapep);
title(' mape plot wrt iterations during training');
%% testing the network
    datan=An(430:630);
    t=length(datan)-7;
     for i=1:t
       tabletest(i,1:7)=datan(i:(i+6));
     end
    ynew=[];
    for i=1:100
     for k=1:t
        sigma=(tabletest(k,1:6))*(w');
        ynew(k)=tanh(sigma);
        errortest(k)=tabletest(k,7)-ynew(k);
        errsqtest(k)=(errortest(k))^2;
        mapetest(k)=abs((errort(k))/(table(k,7)));
     end
     mseptest(i)=((sum(errsqtest))/100);
     mapeptest(i)=((sum(mapetest))/100);
    end
%% plotting results of testing
figure
hold on
plot(tabletest(1:k,7),'r');
plot(ynew,'b');
title(' network output versus original during testing ');
legend('original output','network output');
figure
plot(errortest)
title(' errror of testing ');
figure
plot(1:100,mseptest);
title('rmse plot wrt iterations during testing');
figure
plot(1:100,mapeptest);
title('mape plot wrt iterations during testing');

%% prediction of values
aforp=An;
for i=1:1
    %pv=(atanh(aforp(6,1))-(((aforp(1:5))')*((w(2:6))')))/(w(1,1));
    pv=atanh(aforp(6,1));
    tr1=((aforp(1:5))');
    tr2=((w(2:6))');
    pv=pv-tr1*tr2;
    pv=pv/(w(1,1));
    aforp=[pv ; aforp];
end
figure
plot(1:100,aforp(100:-1:1,1));
title(' predicted from the next date of the latest date in sample ');
