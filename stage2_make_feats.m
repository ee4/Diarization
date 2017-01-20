function [X,t,cfs] = stage2_make_feats(x,fs,fr,wl,B,topt,mvdr,norm_type,decorr_type,F,z_score)
%This generates features X at frame-rate fr for audio time-series x.
%Note that X is of size [F x W], where F is the # of features and W
%is the number of windows (output frames).
%Initially, a time-freq representation (TFR) is made of size [B x W],
%where B is the number of freq bands.
%topt controls the type of transformation matrix H to apply to the raw STFT.
%The change of the spectrum to MVDR (minimum-variance distortionless response)
%can be turned on with mvdr = true [as in Yapanel and Hansen 2008].
%norm_type allows usual log compression ('log'), or power normalization ('PN'),
%or simple PN ('sPN'), as in Kim and Stern [2016].
%decorr_type allows usual cepstral coeefs (dct), or PCA-style decorrelation.
%An exotic decorr_type 'stpca' allows spatiotemporal-PCA decorrelation.
%F is the final # of features (cepstral coeffs or PCA comps) in X.
%z_score = true -> applies z-score to each row of X as final step.
%
%INPUTS:
%%x (double): input time-series [T x 1]
%fs (double): sample-rate of x (samps/sec)
%fr (double): frame-rate of X (frames/sec)
%wl (double): STFT window length in secs
%B (uint): # of freq bands in initial TFR
%topt (str): controls type of transformation (H) to apply to raw STFT
%mvdr (logical): do or don't do MVDR spectral transformation
%norm_type (str): 'log','PN' or 'sPN' normalization
%decorr_type (str): 'CCs' (i.e. dct) or 'PCA' (i.e. eig) or 'STPCA'
%F (uint): % of features
%z_score (logical): z-score rows of X
%
%OUTPUTS:
%X (double): output features [F x W]
%t (double): time line for X in secs [1 x W]
%cfs (double): center-frqs in Hz if needed for plotting [1 x F]
%

%Frame rate and step-size
if nargin<3 || isempty(fr)
  fr = 100; %100 Hz frame-rate
end
stp = fs/fr; %step size (assumes fs is multiple of fr)

%Window and STFT frqs
%I use Kim's settings (25.6-ms wl, and 1024-point nfft) as default,
%because there may be benefit from slightly denser freq spacing.
if nargin<4 || isempty(wl)
  wl = 0.0256; %25.6-ms winlength
end
wl = fix(wl*fs);
if ~mod(wl,2), wl = wl + 1; end
%nfft = 2^nextpow2(wl+1); %try this next time to save compute time
nfft = 2^(nextpow2(wl)+1); %Kim's code for PNCC (most don't do this)
frqs = fs*(0:1/nfft:1/2);
nfrqs = length(frqs);
win = 0.54 - 0.46*cos(2*pi*(0:wl-1)./(wl-1))'; %hamming
wss = 1:stp:length(x)-wl+1; %win start samps
W = length(wss); %# of windows or frames

%Output time-line is delayed by half wl per ASR convention
if nargout>1
  t = (1/fr)*(0:W-1) + 0.5*(wl-1)/fs;
end

%B = # of frqs to keep in initial TFR
if nargin<5 || isempty(B)
  B = 40;
end

%Options for H (transformation matrix)
if nargin<6 || isempty(topt)
  topt = 'male';
end
addpath('C:\Erik\Toolboxes\Aud\Mel_Bark_NFScale');
switch lower(topt)
  case {'male','female'} %(as used by Yapanel and Hansen)
    [H,omegas,cfs] = transform_Hz_to_Warped(fs,nfft,2*B,topt);
    H = H(2:B,:); %remove DC and Nyquist (since would set to 0 anyway)
  case {'gammatone'}
    [H,czs,cfs] = transform_Hz_to_Gammatone(frqs,0,fs/2,B+1);
    H = H.^2; %for gammatone only (according to Kim and Stern [2016])
    H = H(2:B,:); %remove DC and Nyquist (since would set to 0 anyway)
  case {'bark','barks'}
    [H,czs,cfs] = transform_Hz_to_Bark(frqs,0,fs/2,B-1,1,1);
  case {'mel','mels'}
    [H,czs,cfs] = transform_Hz_to_Mel(frqs,0,fs/2,B-1,0,1);
  case {'sgc','sgcs'}
    [H,czs,cfs] = transform_Hz_to_SGCs(frqs,0,fs/2,B-1,1,1);
  otherwise %SGCs scale, but with fixed 1-Bark filters of S & H [1984]
    lowf = 100; highf = 6000;
    [H,czs,cfs] = transform_Hz_to_SGCs_SH1984(frqs,lowf,highf);
    H = H.^2; %like gammatone above, this makes narrower, etc.
    B = size(H,1) + 1;
end


%Compute STFT
X = zeros([B W],'double');
for w = 1:W
  tmp = abs(fft(x(wss(w):wss(w)+wl-1).*win,nfft));
  X(1:B-1,w) = H*(tmp(1:nfrqs).^2);
end

%I add small regularization for stability (others use a floor).
rg = sqrt(eps);
X = X + rg;


%MVDR
if nargin<7 || isempty(mvdr)
  mvdr = true;
end

if mvdr
  
  %IFFT to obtain autocorrelations (real since tiny imag component)
  nfft2 = 2*B;
  X = real(ifft(X([B 1:B-1 B B-1:-1:1],:),nfft2));  
  
  %Calculate MVDR spectrum
  %There is no imag component here, so I don't use conj, etc.
  M = 24; %order
  M1k2 = repmat(M+1-(0:M)',[1 M+1]);
  M1k2 = bsxfun(@minus,M1k2,2*(0:M));
  K2 = (M+1)^2 - sum((1:M));
  asii1 = zeros([K2 1],'uint8');
  asii2 = zeros([K2 1],'uint8');
  M1K2 = sparse(M+1,K2);
  cnt = 0;
  for k = 0:M
    asii1(cnt+1:cnt+M-k+1) = 1:M-k+1;
    asii2(cnt+1:cnt+M-k+1) = k+1:M+1;
    M1K2(k+1,cnt+1:cnt+M-k+1) = M1k2(k+1,1:M-k+1);
    cnt = cnt + M - k + 1;
  end
  as = ones([M+1 1],'double');
  slices = cellslices([M:-1:2 1:M]',M:-1:1,2*M-1:-1:M);
  hii = horzcat(slices{:});
  lam = .001; Rm = 1 + lam*(1:M)'*(1:M); %for regularization
  for w = 1:W
    as(2:M+1) = -(reshape(X(hii,w),[M M]).*Rm)\X(2:M+1,w);
    v = sum(as.*X(1:M+1,w));
    X(1:M+1,w) = (M1K2*(as(asii1).*as(asii2)))./v;
  end
  
  %This is directly after Marple [1987].
  X(M+2:nfft2-M,:) = 0;
  X(nfft2-M+1:nfft2,:) = X(M+1:-1:2,:);
  X = real(fft(X,nfft2)); %only tiny imag components
  X = (1/fs)./X(2:B,:); %this also removes DC and Nyquist
  
  %It isn't possible to go below 0, but does go below rg...
  X = max(X,rg);
  
else
  
  X = X(1:B-1,:);
  
end

B = size(X,1);


%Normalization
if nargin<8 || isempty(norm_type)
  norm_type = 'log';
end
norm_type = lower(norm_type);

if isequal(norm_type,'log')
  
  X = log(X+rg);
  
else

  if isequal(norm_type,'pn')
    
    %Medium-time power (Q), box-car window
    M = 2; %sets integration window
    Q = X + rg;
    for m = 1:M
      Q(:,1:W-m) = Q(:,1:W-m) + X(:,1+m:W);
      Q(:,1+m:W) = Q(:,1+m:W) + X(:,1:W-m);
    end
    for m = 1:M
      Q(:,[m W-m+1]) = Q(:,[m W-m+1])./(M+m);
    end
    Q(:,M+1:W-M+1) = Q(:,M+1:W-M+1)./(2*M+1);
    
    %Asymmetric nonlinear filter (Q0 is Qle)
    lama = 0.999;
    lamb = 0.5; %range 0.25-0.75
    Q0 = zeros([B W],'double');
    Q0(:,1) = 0.9*Q(:,1);
    for w = 2:W
      if Q(:,w)<Q0(:,w-1)
        Q0(:,w) = lamb*Q0(:,w-1) + (1-lamb)*Q(:,w);
      else
        Q0(:,w) = lama*Q0(:,w-1) + (1-lama)*Q(:,w);
      end
    end
    
    %Make excitation/non-excitation decision
    c = 2;
    nonexseg = (Q < c*Q0); %true if non-excitation segment
    
    %Half-wave rectification
    Q0 = max(Q0,0); %Q0 is now Q0
    
    %Noise floor by another asymmetric nonlinear filter
    Qf = zeros([B W],'double');
    Qf(:,1) = 0.9*Q0(:,1);
    for w = 2:W
      if Q0(:,w)<Qf(:,w-1)
        Qf(:,w) = lamb*Qf(:,w-1) + (1-lamb)*Q0(:,w);
      else
        Qf(:,w) = lama*Qf(:,w-1) + (1-lama)*Q0(:,w);
      end
    end
    
    %Temporal masking (Q0 ends as Qtm)
    lamt = 0.85; %lamt<=0.85
    mut = 0.2; %mut<=0.2
    Qp = Q0(:,1);
    for w = 2:W
      Qu = Q0(:,w); %hold for update
      if Qu < lamt*Qp 
        Q0(:,w) = mut*Qp; %here Q0 becomes Qtm
      end
      Qp = max(lamt*Qp,Qu);
    end
    
    %Make R (but still called Q0 here to save RAM)
    Q0 = max(Q0,Qf); %Now Q0 is Q1
    Q0(nonexseg) = Qf(nonexseg); %Now Q0 is R
    
    %Make S (called Q here to save RAM)
    %This is the spectral weight smoothing.
    %They use N=4 for B=40, but say that different B would mean
    %different N, so this my best guess for setting N generally is:
    N = round(B/10);
    l1 = max((1:B)'-N,1);
    l2 = min((1:B)'+N,B);
    lden = 1./(l2 - l1 + 1);
    SS = zeros([B B],'double'); %spectral smoothing transform
    for b = 1:B
      SS(b,l1(b):l2(b)) = lden(b);
    end
    Q = SS*(Q0./Q);
    
    %Modulate feats to make T = P.*S
    X = X.*Q; %X is now T
    
  end
  
  %The following applies to PN and sPN
  %Mean power normalization
  mnpow = (1/B)*sum(X,1);
  %mui = 5.8471e+08, %they initialize by a training database
  mui = median(mnpow); %this is my guess at alternative
  mus = zeros([1 W],'double');
  mus(1) = 0.999*mui + 0.001*mnpow(1);
  for w = 2:W %RC-filter
    mus(w) = 0.999*mus(w-1) + 0.001*mnpow(w);
  end
  mus = mus + sqrt(eps); %my own regularization
  X = bsxfun(@times,X,1./mus);
  
  X = sign(X).*abs(X).^(1/15);
  
end


%Decorrelation to obtain final features
if nargin<9 || isempty(decorr_type)
  decorr_type = 'dct';
end
decorr_type = lower(decorr_type);

if nargin<10 || isempty(F)
  F = min(22,B);
end

switch decorr_type
  case {'cc','ccs','dct'}
    pkg load signal;
    X = dct(X);
    X = X(1:F,:);
    %note: I don't lifter, since usually z-score
  case {'pca','eig','kl'}
    XX = X*X';
    [U,D] = eig(XX+XX');
    X = U(:,B:-1:B-F+1)'*X;
  case {'stpca'}
    L = 5; %since 50-ms is my min segment duration
    L2 = fix(L/2); lags = -L2:L2; %lags
    lwin = 0.54 - 0.46*cos(2*pi*(0:L-1)./(L-1))'; %lag win
    X = repmat(X,[L 1]); %yes, this is RAM-intensive, but much easier
    for l = setdiff(1:L,L2+1)
      brng = (l-1)*B+1:l*B;
      if lags(l)>0, wrng = [lags(l)+1:W 1:lags(l)];
      else wrng = [W+lags(l)+1:W 1:W+lags(l)]; end
      X(brng,:) = lwin(l)*X(brng,wrng);
    end
    XX = X*X';
    [U,D] = eig(XX+XX');
    X = U(:,L*B:-1:L*B-F+1)'*X;
  %otherwise %{'none'}
    %this allows user to look at spectrogram
end


%Optional z-score.
%Note that the first part is the usual mean-normalization.
if nargin<11 || isempty(z_score)
  z_score = true;
end

if z_score
  X = bsxfun(@minus,X,sum(X,2)./W);
  X = bsxfun(@times,X,1./sqrt(sum(X.^2,2)./W));
end
