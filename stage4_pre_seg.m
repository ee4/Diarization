function [SEG,dX] = stage4_pre_seg(X,dopt,lpopt,param,lmin_elim)
%This takes the features X of size [F x W], where F is the # of features,
%and W is the # of windows (time frames), and produces a preliminary
%segmentation (pre_seg), as represented in the sparse logical matrix SEG.
%SEG is of size [P x W], where P is the # of pre-segments.
%
%The method to apply is controled by dopt (raw distance metric option).
%The basic idea of a sliding window BIC is one dopt ('bic'), as is
%Hansen's idea of sliding T2 ('t2'). Since we are only looking for lmaxs,
%measures are equivalent if related by a constant or a monotonic transform,
%so Hansen's T2 is equivalent to the Malhananobis distance, for example.
%I provide my own fast/simple solution, which is abs(smooth(diff(X))) ('dX').
%
%The lpopt can be 'square', which estimates time-varying mus and covs by
%a boxcar low-pass, or 'rc', which estimates time-varying mus and covs by
%a 1st-order RC filter. The param is always in units of samps.
%For 'square', param is the window-length (wl) of the moving-window boxcar
%in samps, i.e the change metric is computed from frames -wl:wl around the
%current frame. For 'rc', param is tau in samps.
%
%Since the whole point of preseg is to quickly identify candidates (with
%lots of false positives), and since wl is too small to allow reliable
%estimation of covariance matrices, diagonal covariance matrices are assumed.
%By whichever method, the segment boundaries are then placed at all
%appropriate local maxima (lmaxs) of the change metric.
%
%The most questionable lmaxs are those which barely emerge from the surrounding
%lmins, so lmin_elim=true will eliminate those lmaxs less than 5% of the
%surrounding lmins (i.e., in the distance metric time-series).
%
%INPUTS:
%X (double): features [F x W]
%dopt (str): option (method) to obtain the raw 'distance' measure
%lpopt (str): option (method) to integrate time-varying mus and covs
%param (uint): window-length for 'square' and tau for 'rc' (in samps)
%lmin_elim (logical): if true, eliminate lmaxs based on comparison to lmins
%
%OUTPUTS:
%SEG (sparse logical): segments matrix [P x W]
%dX (double): distance metric if desired for plotting [1 x W-1]
%

[F,W] = size(X);

if nargin<2 || isempty(dopt)
  dopt = 'diff';
end
dopt = lower(dopt);

if nargin<3 || isempty(lpopt)
  lpopt = 'hamm';
end
lpopt = lower(lpopt);

if nargin<4 || isempty(param)
  param = 5; % wl or tau = 5 samps
  if ismember(dopt,{'abs','dx','diff'})
    param = fix(2*pi*5); %~5 Hz cutoff freq
  end
end


switch lpopt
  case {'rc'}
    a = exp(-1/param);
    b = 1 - a; a = [1 -a];
  case {'box','boxcar','square','moving_win'}
    a = 1;
    b = (1/param)*[ones([1 param])];
  otherwise %{'hamm','hamming','half_cos'}
    bf = 0.54 + 0.46*cos(pi*(0:param-1)./(param-1));
    bf = bf./sum(bf);
    bb = fliplr(bf); a = 1;
end

%Get smoothed estimates of means (mu1, mu2)
mu1 = zeros([F W-1],'double');
mu2 = zeros([F W-1],'double');
if ismember(lpopt,{'rc','tau'})
  for f = 1:F
    mu1(f,:) = filter(b,a,X(f,W-1:-1:1)); %this just to get istate!
    mu1(f,:) = filter(b,a,X(f,1:W-1),mu1(f,end));
    mu2(f,:) = filter(b,a,X(f,W:-1:2));
    mu2(f,W-1:-1:1) = filter(b,a,X(f,W:-1:2),mu2(f,end));
  end
elseif ismember(lpopt,{'box','boxcar','square','moving_win'})
  tmp = filter(b,a,X(:,[param:-1:2 1:W W-1:-1:W-param+1])')';
  mu1 = tmp(:,param:W+param-2);
  mu2 = tmp(:,2*param:W+2*param-2);
%  for f = 1:F
%    tmp = filter(b,a,[X(f,param:-1:2) X(f,:) X(f,W-1:-1:W-param+1)]);
%    mu1(f,:) = tmp(param:W+param-2);
%    mu2(f,:) = tmp(2*param:W+2*param-2);
%  end
else %ismember(lpopt,{'hamm','hamming','half_cos'})
  tmp = filter(bf,a,X(:,[param:-1:2 1:W W-1:-1:W-param+1])')';
  mu1 = tmp(:,param:W+param-2);
  tmp = filter(bb,a,X(:,[param:-1:2 1:W W-1:-1:W-param+1])')';
  mu2 = tmp(:,2*param:W+2*param-2);
end

%Get smoothed estimates of covs (cov1, cov2)
if ~ismember(lower(dopt),{'abs','dx','diff'})
  cov1 = zeros([F W-1],'double');
  cov2 = zeros([F W-1],'double');
  if ismember(lpopt,{'rc','tau'})
    for f = 1:F
      cov1(f,:) = filter(b,a,X(f,W-1:-1:1).^2); %this just to get istate!
      cov1(f,:) = filter(b,a,X(f,1:W-1).^2,cov1(f,end));
      cov2(f,:) = filter(b,a,X(f,W:-1:2).^2);
      cov2(f,W-1:-1:1) = filter(b,a,X(f,W:-1:2).^2,cov2(f,end));
    end
  elseif ismember(lpopt,{'box','boxcar','square','moving_win'})
    for f = 1:F
      %tmp = filter(b,a,[X(f,:).^2 zeros([1 param])]);
      tmp = filter(b,a,[X(f,param:-1:2).^2 X(f,:).^2 X(f,W-1:-1:W-param+1).^2]);
      cov1(f,:) = tmp(param:W+param-2); %tmp(1:W-1);
      cov2(f,:) = tmp(2*param:W+2*param-2); %tmp(1+param:W-1+param);
    end
  else %ismember(lpopt,{'hamm','hamming','half_cos'})
    tmp = filter(bf,a,(X(:,[param:-1:2 1:W W-1:-1:W-param+1]).^2)')';
    cov1 = tmp(:,param:W+param-2);
    tmp = filter(bb,a,(X(:,[param:-1:2 1:W W-1:-1:W-param+1]).^2)')';
    cov2 = tmp(:,2*param:W+2*param-2);
  end
  %cov0 = 0.5*(cov1+cov2) - (0.5*(mu1+mu2)).^2;
  %cov1 = cov1 - mu1.^2; cov2 = cov2 - mu2.^2;
end


%Obtain the time-varying distance metric dX.
%Like 1st-order derivatives, these are assigned to the time-line 0.5:N-0.5,
%and thus dX is of size [1 x W-1].
%The formulas do not include monotonic changes (constants, sqrts, etc.)
%that would have no influence on finding lmaxs.
%The best description of these (other than bic) is Couvreur and Boite [1999].
%These are ordered here from complicated to simple/efficient.
switch lower(dopt)
    
  case {'bic','dbic','ml_ratio'}
    
    cov0 = 0.5*(cov1+cov2) - (0.5*(mu1+mu2)).^2;
    cov1 = cov1 - mu1.^2; cov2 = cov2 - mu2.^2;
    %dX = (2*param+1)*log(prod(cov0)) - param*log(prod(cov1)) - param*log(prod(cov2));
    dX = (2*param+1)*sum(log(cov0)) - param*sum(log(cov1)) - param*sum(log(cov2));
    
    %Only the lmaxs are needed, so dbic can be left as R (max likelihood ratio).
    %However, if using a threshold later:
    %lam = 1; %penalty weight
    %P = .5*(F+.5*F*(F+1))*log(2*wl+1);
    %dbic = 0.5*dbic - 0.5*lam*P;
    %thresh = -1; %recall that dBIC>0 is significant
    %dbic(dbic<thresh) = min(dbic);
  
  case {'bha','bhattacharyya'}
    
    cov1 = cov1 - mu1.^2; cov2 = cov2 - mu2.^2;
    %dX = 0.5*sum(((mu1-mu2).^2)./(cov1+cov2)) + ...
    %     log(prod(cov1+cov2)./(2*sqrt(prod(cov1.*cov2))));
    dX = 0.5*sum(((mu1-mu2).^2)./(cov1+cov2)) + ...
         sum(log(cov1+cov2)) - log((2*sqrt(prod(cov1.*cov2))));
  
  case {'kl','kld','kl_divergence','kullback_leibler'}
    
    cov1 = cov1 - mu1.^2; cov2 = cov2 - mu2.^2;
    dX = sum(((mu1-mu2).^2).*(1./cov1+1./cov2)) + ...
         sum(cov2./cov1) + sum(cov1./cov2);
  
  case {'t2','mah','mahalanobis'}
    
    cov0 = 0.5*(cov1+cov2) - (0.5*(mu1+mu2)).^2;
    dX = sum(((mu1-mu2).^2)./cov0); %Mahalanobis dist is sqrt of this
  
  case {'ahs'} %Bimbot and Mathan [1993], Johnson and Woodland [1998]
    
    cov1 = cov1 - mu1.^2; cov2 = cov2 - mu2.^2;
    dX = log(sum(cov2./cov1).*sum(cov1./cov2));
    %But this doesn't use the mus!
  
  case {'euclid','euclidean'}
    
    dX = sum((mu2-mu1).^2);
   
  otherwise %{'abs','dx','diff'}
    
    dX = sum(abs(mu2-mu1));
    %dX = sum(abs(mu2-mu1).^pw).^(1/pw); %nonlinear avg option
    %Note: dX = exp(sum(log(abs(mu2-mu1))))==prod(abs(mu2-mu1));
     
end


%Local maxima and minima
lmaxs = find(dX(2:W-2)>dX(1:W-3) & dX(2:W-2)>=dX(3:W-1)) + 1;
lmins = find(dX(2:W-2)<dX(1:W-3) & dX(2:W-2)<=dX(3:W-1)) + 1;
while min(diff(lmaxs))<3
  disp('Further filter of dX...');
  dX(2:W-2) = .25*dX(1:W-3) + .5*dX(2:W-2) + .25*dX(3:W-1);
  lmaxs = find(dX(2:W-2)>dX(1:W-3) & dX(2:W-2)>=dX(3:W-1)) + 1;
  lmins = find(dX(2:W-2)<dX(1:W-3) & dX(2:W-2)<=dX(3:W-1)) + 1;
end

if nargin<5 || isempty(lmin_elim)
  lmin_elim = true;
end

%Remove lmaxs less than 5% larger than surrounding lmins
if lmin_elim
  if lmins(1)>lmaxs(1), lmins = [1 lmins]; end
  if lmins(end)<lmaxs(end), lmins = [lmins W-1]; end
  lmins = 0.5*(dX(lmins(1:end-1)) + dX(lmins(2:end)));
  lmaxs = lmaxs(dX(lmaxs)>1.05*lmins);
end


%Make pre-segment matrix SEG of size [P x W]
P = length(lmaxs) + 1; %# of preliminary segments
SEG = logical(sparse([P W]));
SEG(1,1:lmaxs(1)) = true;
for p = 2:P-1
  SEG(p,lmaxs(p-1)+1:lmaxs(p)) = true;
end
SEG(P,lmaxs(P-1)+1:W) = true;
