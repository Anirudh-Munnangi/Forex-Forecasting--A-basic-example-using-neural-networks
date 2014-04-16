function [fit yorigop errorigop]=my_fitness_nor(aorigip,borigip,worigip,corigip,siorigip,tip,wlsip,tableip)
for in=1:tip
        %calculating sigma products before applying basis function 
        for i=1:wlsip
        sigma(i)=worigip(i,1)+(worigip(i,2:8))*((tableip(in,1:7))');
        end
       d=sum((tableip(in,1:7)).^2);
       dsq=sqrt(d);
      %calculation of the psi's
      for i=1:wlsip
          modi(i)=((dsq-borigip(i))/(aorigip(i)));
          inside=(dsq-corigip(i))/(siorigip(i));
          insq=(inside)^2;
          psiorig(i)=((abs((aorigip(i))))^(-1/2))*exp(-1*insq);
      end
      % calculation of the final output
      for i=1:wlsip
          prod(i)=sigma(i)*psiorig(i);
      end
      yorig(in)=sum(prod);
      
       % calculation of error of only llwnn
      errorig(in)=tableip(in,8)-yorig(in);
      errsqorig(in)=(errorig(in))^2;
      mapeorig(in)=((abs((errorig(in))/(tableip(in,8))))*100);
end
   mseporig=sqrt((sum(errsqorig))/tip);
   mapeporig=((sum(mapeorig)))/tip;
   fit=[mseporig mapeporig];
   yorigop=yorig;
   errorigop=errorig;
