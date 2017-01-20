function SEG = stage6_merge_agc(X,SEG,dopt,K)
%This takes the S segments of stage5, and continues the hierarchical
%agglomerative clustering (AGC).
%Currently, only diagonal covariances are supported.
%The AGC stops at K clusters.
%
%INPUTS:
%X (double): features [F x W]
%SEG (sparse logical): initial segments matrix [S x W]
%dopt (str): option (method) to obtain the raw 'distance' measure
%K (uint): final number of clusters
%
%OUTPUTS:
%SEG (sparse logical): segments matrix [K x W]
%

[F,W] = size(X);
[S,W] = size(SEG);
Ns = full(sum(SEG,2))';

if nargin<4 || isempty(K)
  K = 2; %i.e. 2 talkers
end

if nargin<3 || isempty(dopt)
  dopt = 'kl';
end
dopt = lower(dopt);

%Get mus (size is [F x S]).
if ismember(dopt,{'bic','dbic','ml_ratio'})
    sms = X*SEG';
    mus = bsxfun(@times,sms,1./Ns);
else
    mus = X*SEG';
    mus = bsxfun(@times,mus,1./Ns);
end

%Get covs (size is [F x S])
if ~ismember(dopt,{'euclid','euclidean','dx','diff'})
  ssqs = (X.^2)*SEG';
  covs = bsxfun(@times,ssqs,1./Ns) - mus.^2;
end


%Obtain the pair-wise distance metrics (Ds).
Ds = zeros([S S],'double');

switch dopt
    
  case {'bic','dbic','ml_ratio'}
    
    lam = 0.5; %penalty weight
    for s1 = 1:S-1
      ssq0s = bsxfun(@plus,ssqs(:,s1+1:S),ssqs(:,s1));
      sm0s = bsxfun(@plus,sms(:,s1+1:S),sms(:,s1));
      N0s = 1./bsxfun(@plus,Ns(s1+1:S),Ns(s1));
      mu0s = bsxfun(@times,sm0s,N0s);
      cov0s = bsxfun(@times,ssq0s,N0s) - mu0s.^2;
      Ds(s1,s1+1:S) = (Ns(s1+1:S)+Ns(s1)).*log(prod(cov0s)) - ...
                      Ns(s1+1:S).*log(prod(covs(:,s1+1:S))) - ...
                      Ns(s1)*log(prod(covs(:,s1)));
      pens = .5*(F+.5*F*(F+1))*log(Ns(s1+1:S)+Ns(s1));
      Ds(s1,s1+1:S) = 0.5*Ds(s1,s1+1:S) - 0.5*lam*pens;
      %for s2 = s1+1:S
      %  mu0 = (mus(:,[s1 s2])*Ns([s1 s2])')./sum(Ns([s1 s2]));
      %  cov0 = sum(ssqs(:,[s1 s2]),2)./sum(Ns([s1 s2])) - mu0.^2;
      %  Ds(s1,s2) = sum(Ns([s1 s2]))*log(prod(cov0)) - ...
      %        Ns(s1)*log(prod(covs(:,s1))) - Ns(s2)*log(prod(covs(:,s2)));
      %  pen = .5*(F+.5*F*(F+1))*log(sum(Ns([s1 s2])));
      %  Ds(s1,s2) = 0.5*Ds(s1,s2) - 0.5*lam*pen;
      %end
    end
  
  case {'bha','bhattacharyya'}
    
    for s1 = 1:S-1
      Ds(s1,s1+1:S) = 0.5*sum((bsxfun(@minus,mus(:,s1+1:S),mus(:,s1)).^2)./ ...
                               bsxfun(@plus,covs(:,s1+1:S),covs(:,s1))) + ...
                      log(prod(bsxfun(@plus,covs(:,s1+1:S),covs(:,s1)))./ ...
                      (2*sqrt(prod(bsxfun(@times,covs(:,s1+1:S),covs(:,s1))))));
      %for s2 = s1+1:S
      %  Ds = 0.5*sum(((mus(:,s1)-mus(:,s2)).^2)./(covs(:,s1)+covs(:,s2))) + ...
      %       log(prod(covs(:,s1)+covs(:,s2))./(2*sqrt(prod(covs(:,s1).*covs(:,s2)))));
      %end
    end
    %Ds(Ds>0) = 0.5*Ds(Ds>0); %this has no effect here
  
  case {'kl','kld','kl_divergence','kullback_leibler'}
       
    for s1 = 1:S-1
      Ds(s1,s1+1:S) = sum((bsxfun(@minus,mus(:,s1+1:S),mus(:,s1)).^2).* ...
                           bsxfun(@plus,1./covs(:,s1+1:S),1./covs(:,s1)));
      Ds(s1,s1+1:S) = Ds(s1,s1+1:S) + sum(bsxfun(@times,covs(:,s1+1:S),1./covs(:,s1)));
      Ds(s1,s1+1:S) = Ds(s1,s1+1:S) + sum(bsxfun(@times,1./covs(:,s1+1:S),covs(:,s1)));
      %for s2 = s1+1:S
      %  Ds = sum(((mus(:,s1)-mus(:,s2)).^2).*(1./covs(:,s1)+1./covs(:,s2))) ...
      %    + sum(covs(:,s2)./covs(:,s1)) + sum(covs(:,s1)./covs(:,s2));
      %end
    end
    %Ds(Ds>0) = 0.5*Ds(Ds>0) - F; %this has no effect here
    
  case {'t2','mah','mahalanobis'} %Mahalanobis dist is sqrt of this
    
    %I use the Eq. (4) from Couvreur and Boite [1999] (but is this right?)
    for s1 = 1:S-1
      Ds(s1,s1+1:S) = sum((bsxfun(@minus,mus(:,s1+1:S),mus(:,s1)).^2)./ ...
                           bsxfun(@times,covs(:,s1+1:S),covs(:,s1)));
      %for s2 = s1+1:S
      %  Ds = sum(((mus(:,s2)-mus(:,s1)).^2)./(covs(:,s2).*covs(:,s1)));
      %end
    end
    %Ds(Ds>0) = sqrt(Ds(Ds>0)./F); %this has no effect here
    
  
  case {'ahs'} %Bimbot and Mathan [1993], Johnson and Woodland [1998]
    
    %How can this be good; it doesn't use the mus!
    for s1 = 1:S-1
      Ds(s1,s1+1:S) = log(sum(bsxfun(@times,covs(:,s1+1:S),1./covs(:,s1))) ...
                        .*sum(bsxfun(@times,1./covs(:,s1+1:S),covs(:,s1))));
      %for s2 = s1+1:S
      %  Ds = log(sum(covs(:,s2)./covs(:,s1)).*sum(covs(:,s1)./covs(:,s2)));
      %end
    end
    %Ds(Ds>0) = Ds(Ds>0) - 2*log(F); %no effect here
  
  case {'euclid','euclidean'}
    
    for s1 = 1:S-1
      Ds(s1,s1+1:S) = sum(bsxfun(@minus,mus(:,s1+1:S),mus(:,s1)).^2);
      %for s2 = s1+1:S
      %  Ds(s1,s2) = sum((mus(:,s1)-mus(:,s2)).^2);
      %end
    end
   
  otherwise %{'dx','diff'}
    
    for s1 = 1:S-1
      Ds(s1,s1+1:S) = sum(abs(bsxfun(@minus,mus(:,s1+1:S),mus(:,s1))));
      %for s2 = s1+1:S
      %  Ds(s1,s2) = sum(abs((mus(:,s1)-mus(:,s2))));
      %end
    end
     
end

%figure; imagesc(Ds); grid on; colormap(jet); return;

%Ds matrix is easier to work with as a column vector
%of length sum(S-1:-1:1), i.e. the only unique points.
%I also make corresponding col vecs S1s and S2s to index.
%Ds = Ds + Ds';
S1s = repmat((1:S)',[1 S]);
S2s = repmat((1:S),[S 1]);
S1s = S1s(:)'; S2s = S2s(:)'; Ds = Ds(:)';
S1s = S1s(Ds~=0); S2s = S2s(Ds~=0); Ds = Ds(Ds~=0);

nit = S - K;
for it = 1:2%nit
  
  [~,ii] = min(Ds);
  s1 = S1s(ii); s2 = S2s(ii); %s1, s2 are to be merged
  SEG(s1,:) = SEG(s1,:) + SEG(s2,:);
  elims = (S1s==s2 | S2s==s2); %eliminate s2
  Ds = Ds(~elims); S1s = S1s(~elims); S2s = S2s(~elims);
  iis = (S1s==s1 | S2s==s1); %iis, %these require update of Ds
  s1s = S1s(iis); s2s = S2s(iis);
  
  %Update Ns, mus, ssqs and covs
  if ismember(dopt,{'euclid','euclidean','dx','diff'})
    %Update only Ns and mus
    mus(:,s1) = mus(:,[s1 s2]) * Ns([s1 s2])';
    Ns(s1) = sum(Ns([s1 s2]));
    mus(:,s1) = mus(:,s1)./Ns(s1);
  elseif ismember(dopt,{'bic','dbic','ml_ratio'})
    ssqs(:,s1) = sum(ssqs(:,[s1 s2]),2);
    sms(:,s1) = sum(sms(:,[s1 s2]),2);
    Ns(s1) = sum(Ns([s1 s2]));
    mus(:,s1) = sms(:,s1)./Ns(s1);
    covs(:,s1) = ssqs(:,s1)./Ns(s1) - mus(:,s1).^2;
  else
    ssqs(:,s1) = sum(ssqs(:,[s1 s2]),2);
    mus(:,s1) = mus(:,[s1 s2]) * Ns([s1 s2])';
    Ns(s1) = sum(Ns([s1 s2]));
    mus(:,s1) = mus(:,s1)./Ns(s1);
    covs(:,s1) = ssqs(:,s1)./Ns(s1) - mus(:,s1).^2;
  end
  
  %Update Ds
  switch dopt
    case {'bic','dbic','ml_ratio'}
      ssq0s = ssqs(:,s1s) + ssqs(:,s2s);
      sm0s = sms(:,s1s) + sms(:,s2s);
      iN0s = 1./(Ns(s1s)+Ns(s2s));
      mu0s = bsxfun(@times,sm0s,iN0s);
      cov0s = bsxfun(@times,ssq0s,iN0s) - mu0s.^2;
      Ds(iis) = (Ns(s1s)+Ns(s2s)).*log(prod(cov0s)) - ...
                   Ns(s2s).*log(prod(covs(:,s2s))) - ...
                     Ns(s1s).*log(prod(covs(:,s1s)));
      pens = .5*(F+.5*F*(F+1))*log(Ns(s1s)+Ns(s2s));
      Ds(iis) = 0.5*Ds(iis) - 0.5*lam*pens;
    case {'bha','bhattacharyya'}
      Ds(iis) = 0.5*sum(((mus(:,s1s)-mus(:,s2s)).^2)./(covs(:,s1s)+covs(:,s2s))) + ...
                log(prod(covs(:,s2s)+covs(:,s1s)))./(2*sqrt(prod(covs(:,s1s).*covs(:,s2s))));
    case {'kl','kld','kl_divergence','kullback_leibler'}
      Ds(iis) = sum(((mus(:,s1s)-mus(:,s2s)).^2).*(1./covs(:,s1s)+1./covs(:,s2s))) ...
                + sum(covs(:,s2s)./covs(:,s1s)) + sum(covs(:,s1s)./covs(:,s2s));
    case {'t2','mah','mahalanobis'}
      Ds(iis) = sum(((mus(:,s1s)-mus(:,s2s)).^2)./(covs(:,s1s).*covs(:,s2s)));
    case {'euclid','euclidean'}
      Ds(iis) = sum((mus(:,s1s)-mus(:,s2s)).^2);
    otherwise %{'abs','dx','diff'}
      Ds(iis) = sum(abs(mus(:,s1s)-mus(:,s2s)));
  end
  
end

SEG = SEG(unique(S1s),:); %SEG is now size [K x W]
