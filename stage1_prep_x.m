function [x,fs] = stage1_prep_x(x,fs,preemph,spl,downsamp,rng)
%This prepares (or preprocesses) audio file x for feature generations.
%It applies pre-emphasis and raw scaling to ~70 dB SPL (in Pascals) by default.
%Options allow no pre-emphasis, no scaling or different target SPL,
%downsampling by a factor downsamp, and cropping of x to rng.
%Enter 0 for an option to not apply; enter 1 or [] to use default.
%
%INPUTS:
%x [T x 1] (double): input time-series (or enter path-to-file string)
%fs (double): sample-rate of x (don't enter if x is path-to-file string)
%preemph (double): pre-emphasis (0<=preemph<=1) (0 -> don't do) (1 -> default=0.97)
%spl (double): do raw scaling to ~spl dB SPL (0 -> don't do) (1 -> default=70dB)
%downsamp (uint): resample factor (i.e. 2 -> fs/2) (default = don't do)
%rng [1 x 2] (double): allows user to select range of x in secs
%
%OUTPUTS:
%x [T x 1] (double): output time-series
%fs (double): sample-rate of x
%

if ischar(x)
    [x,fs] = audioread(x);
end
x = x(:); T = length(x);

%Typical 1st-order pre-emphasis
if nargin<2 || isempty(preemph) || preemph==1
  preemph = 0.97; %default
end
if preemph>0
  x = filter([1 -preemph],1,x);
end

%RMS scaling and conversion to pressure (Pascals)
%By the way, to change the order of RMS scaling and pre-emphasis,
%just call this script twice (first with spl=0, then with preemph=0).
if nargin>3 && spl>0
  if spl==1 || isempty(spl)
    spl = 70; %target dB SPL
  end
  target_RMS  =20e-6*10^(spl/20);
  wl = round(.2*fs); %RMS averaged over 200-ms window
  W = fix(T/wl);
  rms = sqrt(sum(reshape(x(1:wl*W).^2,[wl W]))./wl);
  rms = [rms sqrt(sum(reshape(x(fix(wl/2):wl*W-fix(wl/2)-1).^2,[wl W-1]))./wl)];
  p90_rms = quantile(rms,0.9);
  scale = target_RMS/p90_rms;
  max_dB_SPL = 85;
  max_RMS  =20e-6*10^(max_dB_SPL/20);
  scalemx = max_RMS/max(rms);
  scale = min(scale,scalemx);
  x = scale*x; %x now in units of Pascals
end

%Optional downsample by factor downsamp
if nargin>4 && ~isempty(downsamp) && downsamp>1
  pkg load signal;
  x = resample(x,1,downsamp);
  fs = fs/downsamp;
end

%Crop to rng (I do this after above, since those are super-fast, and the
%should be done as usual; this rng opt is usually for quick test/display)
if nargin>5 && ~isempty(rng)
  t = (1/fs)*(0:T-1);
  x = x(t>=rng(1) & t<=rng(2));
end
