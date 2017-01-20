function test_diarization_bottom_up_v1(nf,rng)
%This runs through the 6 stages of bottom-up spkr diarization, version 1.
%This script just allows the user to specify various parameters/options,
%to log timing, and plot intermediate results.
%
%INPUTS:
%nf (uint): file number in 1-11 doctor/patient dialogs
%rng (double): time range in seconds to process [1 x 2]
%

prodir = 'C:\Erik\EMRAI\Expts\Prototype\';
audir = 'C:\Erik\Toolboxes\Aud\';
%addpath([audir 'gammatone']);
%addpath('/home/erik/Toolboxes/Erik/Aud/Cepstra');
dadir = 'C:\Erik\EMRAI\Expts\Prototype\data\2016_11\'
didir = [dadir 'Dialogue_Times_EE\'];
wvdir = [dadir 'wav16\'];

if nargin<1 || isempty(nf)
  nf = 4; %the only one that I've marked so far
end

load([didir 'fnames']); fname = fnames{nf},
segname = [didir fnames{nf}(1:end-4) '_segs'],
load(segname); %loads as segments
wavname = [wvdir fnames{nf}(1:end-4) '_cr.wav'],

[x,fs] = audioread(wavname);

if nargin<2 || isempty(rng)
  rng = [0 6]; %I just crop to rng for quick checks
end

segments = segments(segments(:,1)<=rng(2),:);
SCs = size(segments,1); %spkr changes

%%% Stage 1: Prep x %%%
%I do not explore any options for stage 1.
%However, if using GT feats, turn preemph to ~0.5.
disp('Stage 1...'); tic;
x = stage1_prep_x(x,fs,[],[],[],rng);
%x = stage1_prep_x(x,fs,0.5,[],[],rng); %for GT
toc;

plt = 0;
if plt, figure('Position',[1 74 1920 415]);
  t = (1/fs)*(0:numel(x)-1) + rng(1);
  mn = min(x); mx = max(x);
  plot(t,x,'k'); grid on; axis([rng mn mx]);
  for sc = 1:SCs
    line(segments(sc,[1 1]),[mn mx],'Color','r','LineWidth',2);
    text(segments(sc,1),mx*1.05,'SC','Color','r',...
    'HorizontalAlignment','center','VerticalAlignment','bottom');
  end
end
%sound(x,fs); return;


%%% Stage 2: Make features %%%
%I use 4 fixed settings: MFCCs, PNCCs, PN-PMVDR-CCs, and PN-PMVDR
%Within stage2_make_feats.m, consider playing with nfft, MVDR regularization,
%and lowf/highf for default topt (sgccs with 1-Bark filters).
fr = 100;
wl = 0.0256; %0.025, 0.0256
B = 52; %40, 52, 64
F = 13; %13, 22, 40
z_score = true; %true, false
topt = 'mel'; mvdr = 0; ntype = 'log'; dtype = 'CCs'; %MFCCs
%topt = 'gammatone'; mvdr = 0; ntype = 'PN'; dtype = 'CCs'; %PNCCs
%topt = 'male'; mvdr = 1; ntype = 'log'; dtype = 'CCs'; %PMVDR
%topt = 'sgcs'; mvdr = 1; ntype = 'PN'; dtype = 'CCs'; %PN-PMVDR-CCs
%topt = 'sgcs'; mvdr = 1; ntype = 'PN'; dtype = 'STPCA'; %PN-PMVDR-STPCs

%dtype = 'none'; z_score = false; %for plotting spectrogram

disp('Stage 2...'); tic;
[X,t,cfs] = stage2_make_feats(x,fs,fr,wl,B,topt,mvdr,ntype,dtype,F,z_score);
%[X,t,cfs] = stage2_make_feats_GT(x,fs,fr,wl,B+1,topt,mvdr,ntype,dtype,F,z_score);
toc;

plt = 0;
if plt, figure('Position',[1 74 1800 795]);
  B = size(X,1);
  imagesc(t+rng(1),1:B,X(B:-1:1,:));
  colormap(jet); grid on;
  set(gca,'YTick',[1 1+fix(B/4) 1+ fix(B/2) 1+fix(3*B/4) B]);
  set(gca,'YtickLabel',round(cfs([B B-fix(B/4) B-fix(B/2) B-fix(3*B/4) 1])));
  xlabel('secs'); ylabel('Hz');
  for sc = 1:SCs
    line(segments(sc,[1 1]),[0 B+1],'Color','k','LineWidth',2);
    text(segments(sc,1),0,'SC','Color','k',...
    'HorizontalAlignment','center','VerticalAlignment','bottom');
  end
end
%return;


%%% Stage 3: SAD_VAD %%%



%%% Stage 4: Pre-segmentation %%%

dopt = 'diff'; %'diff','euclid'
dopt = 'bic'; %,'kl','bah','bic'
lpopt = 'hamm'; %'box','hamm','rc'
param = 5; %5, 10, 15
lmin_elim = false; %true, false

disp('Stage 4...'); tic;
[SEG,dX] = stage4_pre_seg(X,dopt,lpopt,param,lmin_elim);
toc;

plt = 0;
if plt
  %figure; imagesc(SEG); grid on; return;
  
  figure('Position',[1 74 1800 795]);
  subplot(3,1,1:2);
  [B,W] = size(X);
  imagesc(t+rng(1),1:B,X(B:-1:1,:));
  colormap(jet); grid on;
  set(gca,'YTick',[1 1+fix(B/4) 1+ fix(B/2) 1+fix(3*B/4) B]);
  set(gca,'YtickLabel',round(cfs([B B-fix(B/4) B-fix(B/2) B-fix(3*B/4) 1])));
  xlabel('secs'); ylabel('Hz');
  [P,W] = size(SEG);
  for p = 1:P
    ons = find(SEG(p,:),1,'first'); offs = find(SEG(p,:),1,'last');
    line(t([ons ons])+rng(1)-.5/fr,[0 B+1],'Color','k','LineWidth',2);
    line(t([offs offs])+rng(1)+.5/fr,[0 B+1],'Color','k','LineWidth',2);
  end
  for sc = 1:SCs
    line(segments(sc,[1 1]),[0 B+1],'Color','r','LineWidth',2);
    text(segments(sc,1),0,'SC','Color','r',...
    'HorizontalAlignment','center','VerticalAlignment','bottom');
  end
  
  
  subplot(3,1,3);
  plot(t(1:W-1)+rng(1)+.5/fr,dX,'r'); grid on; hold on;
  mn = min(dX); mx = max(dX);
  axis([t(1)+rng(1)+.5/fr t(W-1)+rng(1)+.5/fr mn mx]);
  xts = [];
  for p = 1:P
    ons = find(SEG(p,:),1,'first'); offs = find(SEG(p,:),1,'last');
    xts = [xts t(ons)+rng(1)-.5/fr];
    %line(t([ons ons])+rng(1)+.5/fr,[mn mx],'Color','r','LineWidth',2);
    %line(t([offs offs])+rng(1)+.5/fr,[mn mx],'Color','b','LineWidth',2);
  end
  set(gca,'XTick',xts);
  for sc = 1:SCs
    line(segments(sc,[1 1]),[mn mx],'Color','r','LineWidth',2);
    text(segments(sc,1),mx*1.05,'SC','Color','r',...
    'HorizontalAlignment','center','VerticalAlignment','bottom');
  end
end
%return;


%%% Stage 5: Merge Adjacent %%%

%dopt = 'abs';
%dopt = 'mah';
%dopt = 'kl';
dopt = 'bic';
S_fraction = 0.75;

disp('Stage 5...'); tic;
SEG5 = stage5_merge_adjacent(X,SEG,dopt,S_fraction);
toc;

plt = 1;
if plt, figure('Position',[1 74 1920 915]);
  t = t + rng(1); [B,W] = size(X);
  [P,W] = size(SEG); [S,W] = size(SEG5);
  W3 = fix(W/3);
  for s = 1:3, subplot(3,1,s);
    wrng = (s-1)*W3+1:s*W3;
    imagesc(t(wrng),1:B,X(B:-1:1,wrng));
    colormap(jet); %grid on;
    set(gca,'YTick',[1 1+fix(B/4) 1+ fix(B/2) 1+fix(3*B/4) B]);
    set(gca,'YtickLabel',round(cfs([B B-fix(B/4) B-fix(B/2) B-fix(3*B/4) 1])));
    xlabel('secs'); ylabel('Hz');
    for p = 1:P
      ons = find(SEG(p,:),1,'first'); offs = find(SEG(p,:),1,'last');
      line(t([ons ons])-.45/fr,[0 B+1],'Color','g','LineWidth',1);
      line(t([offs offs])+.45/fr,[0 B+1],'Color','k','LineWidth',1);
    end
    for s = 1:S
      ons = find(SEG5(s,:),1,'first'); offs = find(SEG5(s,:),1,'last');
      line(t([ons ons])-.45/fr,[0 B+1],'Color','k','LineWidth',2);
      line(t([offs offs])+.45/fr,[0 B+1],'Color','b','LineWidth',2);
    end
    
    for sc = 1:SCs
      line(segments(sc,[1 1]),[0 B+1],'Color','r','LineWidth',2,'LineStyle','--');
      text(segments(sc,1),0,'SC','Color','r',...
        'HorizontalAlignment','center','VerticalAlignment','bottom');
    end
    
  end
  
end



%%% Stage 6: Merge by agglomerative clustering (AGC) %%%

dopt = 'bic'; %dopt = 'kl'; %dopt = 'mah'; %dopt = 'euclid';
K = 3; %2, 3 (final # of clusters)

disp('Stage 6...'); tic;
SEG6 = stage6_merge_agc(X,SEG5,dopt,K);
toc;

plt = 0;
if plt, figure('Position',[1 74 1920 915]);
  
  
  
end

return;

