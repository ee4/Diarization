function SEG = stage5_merge_adjacent(X,SEG,dopt,S)
%v2 is to test if it is faster to keep a sums-of-squares (ssqs) matrix.
%
%This takes the raw segments of stage4, which are still too many to
%aply to computationally-expensive methods, and merges adjacent
%segments. It proceeds just like hierarchical agglomerative clustering,
%except that only neighboring segments can be merged to save time.
%Also, as with stag4, only diagonal covariances are allowed.
%An initial stage goes through the smallest segments (2-5 samps) and
%forces them to merge with one or the other neighboring segment.
%The second stage continues until a reduced # of segments as specified
%by S is reached.
%
%INPUTS:
%X (double): features [F x W]
%SEG (sparse logical): prelim segments matrix [P x W]
%dopt (str): option (method) to obtain the raw 'distance' measure
%S (number): if 0 < S < 1: reduce to fix(S*P) segments
%                if S < 0: reduce to P + S segments
%                if S > 1: reduce to S segments
%
%OUTPUTS:
%SEG (sparse logical): segments matrix [S x W]
%

[F,W] = size(X);
[P,W] = size(SEG);
Ns = full(sum(SEG,2))';

%Set S, the final # of segments after merging
if nargin<4 || isempty(S)
  S = 0.5; %reduce # of segments by half
end
if S<0, S = P + fix(S);
elseif S<1, S = fix(S*P);
end

if nargin<3 || isempty(dopt)
  dopt = 'bic';
end
dopt = lower(dopt);

%Get mus (size is [F x P]).
mus = X*SEG';
mus = bsxfun(@times,mus,1./Ns);

%Get ssqs and covs
if ~ismember(dopt,{'euclid','euclidean','dx','diff'})
  ssqs = (X.^2)*SEG';
  covs = bsxfun(@times,ssqs,1./Ns) - mus.^2;
end

%Obtain the pair-wise distance metrics (Ds).
%The best description of these (other than BIC) is Couvreur and Boite [1999].
%These are ordered here from complicated to simple/efficient.
switch dopt
    
  case {'bic','dbic','ml_ratio'}
    
    cov0s = ssqs(:,1:P-1) + ssqs(:,2:P);
    cov0s = bsxfun(@times,cov0s,1./(Ns(1:P-1)+Ns(2:P)));
    mu0s = bsxfun(@times,mus(:,1:P-1),Ns(1:P-1)) + ...
           bsxfun(@times,mus(:,2:P),Ns(2:P));
    mu0s = bsxfun(@times,mu0s,1./(Ns(1:P-1)+Ns(2:P)));
    cov0s = cov0s - mu0s.^2;
    Ds = (Ns(1:P-1)+Ns(2:P)).*log(prod(cov0s)) - ...
          Ns(1:P-1).*log(prod(covs(:,1:P-1))) - Ns(2:P).*log(prod(covs(:,2:P)));
    lam = .05; %penalty weight (usually 1, but I find .5 sufficient)
    pen = .5*(F+.5*F*(F+1))*log(Ns(1:P-1)+Ns(2:P));
    Ds = 0.5*Ds - 0.5*lam*pen;
  
  case {'bha','bhattacharyya'}
    
    Ds = 0.5*sum(((mus(:,1:P-1)-mus(:,2:P)).^2)./(covs(:,1:P-1)+covs(:,2:P))) + ...
         log(prod(covs(:,1:P-1)+covs(:,2:P))./(2*sqrt(prod(covs(:,1:P-1).*covs(:,2:P)))));
    %Ds = 0.5*Ds; %this has no effect here
  
  case {'kl','kld','kl_divergence','kullback_leibler'}
    
    Ds = sum(((mus(:,1:P-1)-mus(:,2:P)).^2).*(1./covs(:,1:P-1)+1./covs(:,2:P))) ...
          + sum(covs(:,2:P)./covs(:,1:P-1)) + sum(covs(:,1:P-1)./covs(:,2:P));
    %Ds = 0.5*Ds - F; %this has no effect here
    
  case {'t2','mah','mahalanobis'} %Mahalanobis dist is sqrt of this
    
    %This is suposed to be fast, but computation of cov0s
    %turns out to be harder than covs for each segment.
    %cov0s = covs(:,1:P-1) + covs(:,2:P);
    %cov0s = bsxfun(@times,cov0s,1./(Ns(1:P-1)+Ns(2:P)));
    %mu0s = bsxfun(@times,mus(:,1:P-1),Ns(1:P-1)) + ...
    %       bsxfun(@times,mus(:,2:P),Ns(2:P));
    %mu0s = bsxfun(@times,mu0s,1./(Ns(1:P-1)+Ns(2:P)));
    %cov0s = cov0s - mu0s.^2;
    %Ds = sum(((mus(:,1:P-1)-mus(:,2:P)).^2)./cov0s;
    
    %So, I use the Eq. (4) from Couvreur and Boite [1999] (but is this right?)
    Ds = sum(((mus(:,2:P)-mus(:,1:P-1)).^2)./(covs(:,2:P).*covs(:,1:P-1)));
    %Ds = sqrt(Ds./F); %this has no effect here
    
  
  case {'ahs'} %Bimbot and Mathan [1993], Johnson and Woodland [1998]
    
    %How can this be good; it doesn't use the mus!
    Ds = log(sum(covs(:,2:P)./covs(:,1:P-1)).*sum(covs(:,1:P-1)./covs(:,2:P)));
    %Ds = Ds - 2*log(F); %no effect here
  
  case {'euclid','euclidean'}
    
    Ds = sum((mus(:,1:P-1)-mus(:,2:P)).^2);
   
  otherwise %{'dx','diff'}
    
    Ds = sum(abs(mus(:,1:P-1)-mus(:,2:P)));
    %Ds = sum(abs(mus(:,1:P-1)-mus(:,2:P)).^pw).^(1/pw); %nonlinear avg option
     
end


plt = 0;
if plt
  Ds1 = sum(abs(mus(:,1:P-1)-mus(:,2:P)).^2); %euclid
  Ds2 = sum(((mus(:,2:P)-mus(:,1:P-1)).^2)./(covs(:,2:P).*covs(:,1:P-1))); %mah
  Ds3 = sum(((mus(:,1:P-1)-mus(:,2:P)).^2).*(1./covs(:,1:P-1)+1./covs(:,2:P))) ...
            + sum(covs(:,2:P)./covs(:,1:P-1)) + sum(covs(:,1:P-1)./covs(:,2:P)); %kl
  Ds4 = 0.5*sum(((mus(:,1:P-1)-mus(:,2:P)).^2)./(covs(:,1:P-1)+covs(:,2:P))) + ...
           log(prod(covs(:,1:P-1)+covs(:,2:P))./(2*sqrt(prod(covs(:,1:P-1).*covs(:,2:P))))); %bah
  Ds5 = Ds; %use 'bic' as dopt
  figure('Position',[1 74 1920 915]);
  labs = {'euclidean','mahalanobis','kl divergence','bhattacharyya','bic'};
  for r = 1:5
    for c = setdiff(1:5,r), subplot(5,5,(c-1)*5+r);
      eval(['scatter(Ds' int2str(r) ',Ds' int2str(c) ');']);
      xlabel(labs{r}); ylabel(labs{c}); grid on;
    end
  end
end


%As lists the active segments out of the original P.
As = 1:P;

nit = P - S;
for it = 1:nit
  
  [~,ii] = min(Ds(As(1:end-1)));
  p = As(ii); %seg # in 1:P to merge to
  pp1 = As(ii+1); %"p plus 1"
  pm1 = As(max(1,ii-1)); %"p minus 1"
  
  As = setdiff(As,pp1);
  SEG(p,:) = SEG(p,:) + SEG(pp1,:);
  
  %Update Ns, mus, ssqs and covs
  if ismember(dopt,{'euclid','euclidean','dx','diff'})
    %Update only Ns and mus
    mus(:,p) = mus(:,[p pp1]) * Ns([p pp1])';
    Ns(p) = sum(Ns([p pp1]));
    mus(:,p) = mus(:,p)./Ns(p);
  else
    ssqs(:,p) = sum(ssqs(:,[p pp1]),2);
    mus(:,p) = mus(:,[p pp1]) * Ns([p pp1])';
    Ns(p) = sum(Ns([p pp1]));
    mus(:,p) = mus(:,p)./Ns(p);
    covs(:,p) = ssqs(:,p)./Ns(p) - mus(:,p).^2;
  end
  
  %Update Ds
  switch dopt
    case {'bic','dbic','ml_ratio'}
      
      ssq0 = sum(ssqs(:,[pm1 p]),2);
      mu0 = mus(:,[pm1 p]) * Ns([pm1 p])';
      N0 = sum(Ns([pm1 p])); mu0 = mu0./N0;
      cov0 = ssq0./N0 - mu0.^2;
      Ds(pm1) = (Ns(pm1)+Ns(p)).*log(prod(cov0)) - ...
                 Ns(pm1).*log(prod(covs(:,pm1))) - Ns(p).*log(prod(covs(:,p)));
      pen = .5*(F+.5*F*(F+1))*log(Ns(pm1)+Ns(p));
      Ds(pm1) = 0.5*Ds(pm1) - 0.5*lam*pen;
      
      ssq0 = sum(ssqs(:,[p pp1]),2);
      mu0 = mus(:,[p pp1]) * Ns([p pp1])';
      N0 = sum(Ns([p pp1])); mu0 = mu0./N0;
      cov0 = ssq0./N0 - mu0.^2;
      Ds(p) = (Ns(p)+Ns(pp1)).*log(prod(cov0)) - ...
               Ns(p).*log(prod(covs(:,p))) - Ns(pp1).*log(prod(covs(:,pp1)));
      pen = .5*(F+.5*F*(F+1))*log(Ns(p)+Ns(pp1));
      Ds(p) = 0.5*Ds(p) - 0.5*lam*pen;
      
    case {'bha','bhattacharyya'}
      Ds(pm1) = 0.5*sum((mus(:,pm1)-mus(:,p)).^2)./(covs(:,pm1)+covs(:,p)) + ...
           log(prod(covs(:,pm1)+covs(:,p))./(2*sqrt(prod(covs(:,pm1).*covs(:,p)))));
      Ds(p) = 0.5*sum((mus(:,p)-mus(:,pp1)).^2)./(covs(:,p)+covs(:,pp1)) + ...
           log(prod(covs(:,p)+covs(:,pp1))./(2*sqrt(prod(covs(:,p).*covs(:,pp1)))));
    case {'kl','kld','kl_divergence','kullback_leibler'}
      Ds(pm1) = sum(((mus(:,pm1)-mus(:,p)).^2).*(1./covs(:,pm1)+1./covs(:,p))) ...
            + sum(covs(:,p)./covs(:,pm1)) + sum(covs(:,pm1)./covs(:,p));
      Ds(p) = sum(((mus(:,p)-mus(:,pp1)).^2).*(1./covs(:,p)+1./covs(:,pp1))) ...
            + sum(covs(:,pp1)./covs(:,p)) + sum(covs(:,p)./covs(:,pp1));
    case {'t2','mah','mahalanobis'}
      Ds(pm1) = sum(((mus(:,p)-mus(:,pm1)).^2)./(covs(:,p).*covs(:,pm1)));
      Ds(p) = sum(((mus(:,pp1)-mus(:,p)).^2)./(covs(:,pp1).*covs(:,p)));
    case {'ahs'}
      Ds(pm1) = log(sum(covs(:,p)./covs(:,pm1))*sum(covs(:,pm1)./covs(:,p)));
      Ds(p) = log(sum(covs(:,pp1)./covs(:,p))*sum(covs(:,p)./covs(:,pp1)));
    case {'euclid','euclidean'}
      Ds(pm1) = sum((mus(:,pm1)-mus(:,p)).^2);
      Ds(p) = sum((mus(:,p)-mus(:,pp1)).^2);
    otherwise %{'dx','diff'}
      Ds(pm1) = sum(abs(mus(:,pm1)-mus(:,p)));
      Ds(p) = sum(abs(mus(:,p)-mus(:,pp1)));
  end
  
end

SEG = SEG(As,:); %SEG is now size [S x W]
