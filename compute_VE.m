clear VE_family_matching
load('CC.mat')%The serial number of the statistical area
    for l=2:10
        l
        if l<9
            load('Temp_time_vac_rand_a.mat')
        else
            load('Temp_time_vac_rand_b.mat')
        end
        
        
        T1=reshape(temp1(:,l,:),1e6*10,1);%which person was infected    
        T2=reshape(temp2(:,l,:),1e6*10,1); %which day
        
        %%%
        % For the case of SI:
        
        clear Family
        w1=(T1(T1>0));
        w2=(T2(T1>0));
        [gg,ii]=sort(w1(:,1));%Sort the people according to their  serial number

        a=abs(w2(ii)-smoothdata(w2(ii),"movmedian",4))<4; %for the case of SI, find household ifections within 4 days
        b=w1(ii);
        c=w2(ii);
        Family(:,1)=b(a);
        Family(:,2)=c(a);
        Family(:,3)=t_rand(Family(:,1),l);%day of receiving the second dose 

        Family(:,4)=Family(:,3)-Family(:,2);
        %%%
        
        for n=70:1:120
            J=0;
            clear X_non X_vac
            for s=2:1579 %matchig statistical areas
                if sum(ismember(Family(:,1),CC(s-1):CC(s)))>0
                    x=find(ismember(Family(:,1),CC(s-1):CC(s)));
                    temp_c=Family(x,:); 
                    % For the case of limitations:
                    %taking only families that all the people are protected,
                    %i.e they got the second dose more than 7 days ago)       
                    x1=find(smoothdata(temp_c(:,3)<n-7,'movmean',5)>=0.5);
                    a=find(temp_c(:,3)<=n-7);
                    b=find(temp_c(:,3)>=21+n);
                    b1=intersect(b,x1);
                    a1=intersect(a,x1);
                    
                    a1=a1(randperm(length(a1)));
                    
                    b1=b1(randperm(length(b1)));
                    
                    
                    
                    %For each statistical area,
                    %take the same  number of vaccinated and unvaccinated people 
                    
                    
                    c=min([length(a1),length(b1)]);
                    if c>0
                        X_vac(J+1:J+c)=x(a(1:c));
                        X_non(J+1:J+c)=x(b(1:c));
                        J=J+c;
                    end
                    
                end
            end
            VE(n,l)=sum(Family(X_vac,2)<=n&Family(X_vac,2)>=n-7)/sum(Family(X_non,2)<=n&Family(X_non,2)>=n-7);
          end
     end

