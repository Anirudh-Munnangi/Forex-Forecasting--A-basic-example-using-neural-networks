%% INITIALISATION
clear all,close all;clc
P=importdata('junetrain.csv');
%P=importdata('julytrain.csv');
%P=importdata('augtrain.csv');
%P=importdata('septrain.csv');
sumF=0;
gmax=1000;
N=20;
pm=0.003;
CR=0.5;
itc=floor(((CR*N)/2)-1);
%its=(N-(CR*N));
its=(N-(itc*2))-2;
Q=70;
lb=-2;
ub=2;
nf=7;
nh=5;
no=1;
wih=nf*nh;
who=nh*no;
tw=wih+who;
lc=4*tw;
T=P(:,nf+1);
%% INITIAL POPULATION GENERATION
for pol=1:N
A=randi([1000,9999],1,tw);
ctr=1;
for n=1:tw
    X=A(n);
    S(ctr)=floor(X/1000);
    ctr=ctr+1;
    S(ctr)=mod(floor(X/100),10);
    ctr=ctr+1;
    S(ctr)=mod(floor(X/10),10);
    ctr=ctr+1;
    S(ctr)=mod(X,10);
    ctr=ctr+1;   
end
    D=S;
    K(:,1:length(D),pol)=D;
    Kh(:,1:length(S),pol)=K(:,:,pol);
 
%% WEIGHT------EXTRACTION 
for i=1:tw
 r=A(i);
 f=floor(r/(10^(floor(log10(r)))));
 if(f>=0 && f<5)
     W=-(r-(f*(10^3)))/(10^2); 
 elseif(f>=5 && f<=9) 
      W=(r-(f*(10^3)))/(10^2);     
 end
 B(i)=W;
end
 reshape(B(1:wih),nf,nh);
 reshape(B(wih+1:tw),nh,no);
 %% IMPLEMENTATION OF BACK-PROPAGATION ALGORITHM 
sumerr=0;
  for j=1:Q
        II=P(j,1:nf)';
        OI=II;
        IH=(reshape(B(1:wih),nf,nh))'*OI;
        OH=sigmoid(IH);
        IOp=(reshape(B(wih+1:tw),nh,no))'*OH;
        OOp=sigmoid(IOp);
        err=(T(j,1)-OOp)^2;
        sumerr=sumerr+err;
  end      
   rmse=sqrt(sumerr/Q);
   mse=sumerr/Q;
   chr(:,:,pol)=int2str(A);
   F=1/rmse;
   Z(pol)=F;
   %Z(pol)=rmse;
end
for gen=1:gmax
%% -------ELITISM----------------
% sf=sort(Z,'descend');
% for ctr=1:its
%     floc(ctr)=find(Z==sf(ctr),1,'first'); 
%     Ks(:,:,ctr)=Kh(:,:,floc(ctr));
% end
%% TOURNAMENT SELECTION
% for ctr=1:its
% selp=ceil(N*rand(1,2));
% pospc1=selp(1);
% pospc2=selp(2);
% p=Kh(:,:,selp);
% F1=Z(pospc1);
% F2=Z(pospc2);
%  if(F1>F2)
%      pc=Kh(:,:,pospc1);
%  else
%      pc=Kh(:,:,pospc2);
%  end
%  Ks(:,:,ctr)=pc;
% end
%% FORMATION OF MATING POOL USING "ROULETTE WHEEL SELECTION"
 accumulation =cumsum(Z);
 for in=1:its
  p = rand() * accumulation(end);
  chosen_index = -1;
  for index = 1 : length(accumulation)
    if (accumulation(index)> p)
      chosen_index = index;
      break;
    end
  end
  choice(in) = chosen_index;
  Ks(:,:,in)=Kh(:,:,choice(in));
 end
%% ------TWO-POINT CROSSOVER-----------
 for itr=0:itc
  selposp=ceil(its*rand(1,2)); 
  selposc=ceil(lc*rand(1,2));
  posp1=selposp(1);
  posp2=selposp(2);
  sposc=selposc(1);
  eposc=selposc(2);
  p1=Ks(:,:,selposp(1));
  p2=Ks(:,:,selposp(2));
  temp=p1(1,sposc:eposc,1);
  p1(:,sposc:eposc,1)=p2(:,sposc:eposc,1);
  p2(:,sposc:eposc,1)=temp;
  Kc(:,:,(2*itr+1))=p1;
  Kc(:,:,2*(itr+1))=p2;
 end
newp=[Ks;Kc];
%% --------MUTATION-------------
  K1=reshape(newp,1,lc*N);
  nbits=round(pm*N*lc);
  bitpos=randi([1,lc*N],1,nbits);
  for b=1:nbits
  if(K1(bitpos(b))>5)
      K1(bitpos(b))=K1(bitpos(b))-1;
  else
      K1(bitpos(b))=K1(bitpos(b))+1;
  end
  end 
 ctr=0;
for i=1:20
Kh(1,1:lc,i)=K1(ctr+1:ctr+lc);     %%%NEW POPULATION%%%
ctr=ctr+lc;
end
%% WEIGHT------EXTRACTION 
 for pol=1:N
  ctr1=1;
  for h=1:tw
     di(h,:,pol)=Kh(1,ctr1:ctr1+3,pol); % extraction of digits in rows of four
     fd(1,1:tw,pol)=Kh(1,1:4:lc,pol);
  ctr1=ctr1+4;
  end
 end
 % extraction of first digits in the above rows of four
 
% finding weights 
 for pol=1:N
    for n=1:tw
        val=fd(:,tw,pol);
        r=di(n,1,pol)*1000+di(n,2,pol)*100+di(n,3,pol)*10+di(n,4,pol);
        if(val<5)
            inval=-1*(r-val*1000)/1000;
        else
            inval=(r-val*1000)/1000;
        end
      extw(:,n,pol)=inval;
    end  
 end
 
  %% IMPLEMENTATION OF BACK-PROPAGATION ALGORITHM 
for pol=1:N
  sumerr1=0;
  for j=1:Q
        II1=P(j,1:nf)';
        OI1=II1;
        IH1=reshape(extw(:,1:wih,pol),nf,nh)'*OI1;
        OH1=sigmoid(IH1);
        IOp1=reshape(extw(:,wih+1:tw,pol),nh,no)'*OH1;
        OOp1=sigmoid(IOp1);
        err1=0.5*(T(j,1)-OOp1)^2;
        sumerr1=sumerr1+err1;
  end
     rmse1=sqrt(sumerr1/Q);
     mse1=sumerr1/Q;
     F1=1/rmse1;
     Z(pol)=F1; 
end
   Arr(gen)=max(Z);
   Arrloc=find(Z==Arr(gen),1,'first');
   chrmF(:,:,gen)=Kh(:,:,Arrloc);
   SFit(gen)=sum(Z);
   avg(gen)=SFit(gen)/N;
end
Arrmax=max(Arr);
Arrmaxl=find(Arr==Arrmax);
chrmmF=chrmF(:,:,Arrmaxl);
%% PLOTTING GRAPHS 
figure
hold on
plot(1:gmax,Arr,'--r','LineWidth',2)
%plot(1:gmax,avg,'--b','LineWidth',2)
xlabel('No of Generation');
ylabel('Fitness');
% legend(' best ', ' average ');

% %% WEIGHT------EXTRACTION 
%   ctr1=1;
%   for h=1:tw
%     di(h,:,1)=chrmmF(1,ctr1:ctr1+3,1); % extraction of digits in rows of four
%   ctr1=ctr1+4;
%   end
%  
% % extraction of first digits in the above rows of four
% fd(1,1:tw,1)=chrmmF(1,1:4:lc,1);
% % finding weights :)
%  
%     for n=1:tw
%         val=fd(1,n,1);
%         r=di(n,1,1)*1000+di(n,2,1)*100+di(n,3,1)*10+di(n,4,1);
%         if(val<5)
%             inval=-1*(r-fd(1,n,1)*1000)/1000;
%         else
%             inval=(r-fd(1,n,1)*1000)/1000;
%         end
%       extwn(1,n,1)=inval;
%     end  
% V=extwn(1,1:wih,1);
% W=extwn(1,wih+1:tw,1);
% w1=reshape(V(:,:,1),nf,nh);
% w2=reshape(W(:,:,1),nh,no);
% %% -----------PREDICTION------------------ 
% B=importdata('junetest.csv'); 
% Tsumerr=0; 
% R=30;
% Acd=B(:,8);
% for p=1:R
%       TII=B(p,1:7)';
%       TOI=TII;
%       TIH=w1'*TOI;
%       TOH=sigmoid(TIH);
%       TIO=w2'*TOH;
%       TOO=sigmoid(TIO);
%       Terr= 0.5*(B(p,8)-TOO)^2;
%       Tsumerr=Tsumerr+Terr; 
%       Pd(p)=TOO;
% end
%  
%    Trmse=sqrt(Tsumerr/R);
%    figure
%    hold on
%    plot(1:R,Acd,'--r','LineWidth',2)
%    plot(1:R,Pd,'--k','LineWidth',2)
%    xlabel('No of Years ');
%    ylabel('Rainfall');
%    legend(' Actual data ', ' predicted data ');
%  
%  
 
