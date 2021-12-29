%RNN_GLM_scan

clear
clc

% cd ..; setpaths_GLMspiketools;
set(0,'DefaultLineLineWidth',1,...
    'DefaultLineMarkerSize',8, ...
    'DefaultAxesLineWidth',1, ...
    'DefaultAxesFontSize',14,...
    'DefaultAxesFontWeight','Bold');

%%  ===== Set parameters for simulating a GLM  ============ %

dt = .001; % Bin size for simulating model & computing likelihood (in units of stimulus frames)
dt = .001;  % Bin size for simulating model & computing likelihood (must evenly divide dtStim);
nkt = 20;    % Number of time bins in stimulus filter k %% FIX THIS SO IT DOESN'T MAKE NAN
gg = makeSimStruct_GLM(nkt,dt,dt);  % Create GLM struct with default params

%% 
%Initialize kernel methods
% Make basis for self-coupling term
nb = 4;
ihbasprs.ncols = nb; % number of basis vectors
ihbasprs.hpeaks = [.002, .005]; % peak of 1st and last basis vector
ihbasprs.b = .001;  % scaling (smaller -> more logarithmic scaling)
ihbasprs.absref = .004; % absolute refractory period basis vector (optional)
% Make basis 
[iht,ihbas,ihbasis] = makeBasis_PostSpike(ihbasprs,dt);
nht = length(iht); % number of bins

% Make basis for cross-coupling term
ihbasprs2.ncols = nb;  % number of basis vectors
ihbasprs2.hpeaks = [0.001,.005]; % put peak at 10ms and "effective" 1st peak at 0
ihbasprs2.b = .002;  % smaller -> more logarithmic scaling
ihbasprs2.absref = []; % no abs-refracotry period for this one
% Make basis
[iht2,ihbas2,ihbasis2] = makeBasis_PostSpike(ihbasprs2,dt);
nht2 = length(iht2);

% pad to put them into the same time bins, if necessary
if nht2>nht
    % padd ih1 with zeros
    iht = iht2; zz = zeros(nht2-nht,ihbasprs.ncols);
    ihbas = [ihbas;zz]; ihbasis = [ihbasis;zz]; nht=nht2;
elseif nht2<nht
    % padd ih1 with zeros
    iht2 = iht; zz = zeros(nht-nht2,ihbasprs2.ncols);
    ihbas2 = [ihbas2;zz]; ihbasis2 = [ihbasis2;zz]; nht2=nht;
end  

%% Set self-coupling weights
wself = [-5; 1; .2; -.15]; % weights for self-coupling term  randn(nb,1); %
ihself = ihbasis*wself; % self-coupling filter
wcpl = [0.2; 0.8; 0.5; .9];%abs(randn(nb,1)); %0.5; % weights for cross-coupling term
ihcpl = ihbasis2*wcpl; % cross-coupling filter
% ihself = flip(ihself);
% ihcpl = flip(ihcpl);
clf; plot(iht, exp(ihself), iht, exp(ihcpl), iht, iht*0+1, 'k--');
legend('self-coupling', 'cross-coupling');
xlabel('time lag (s)');
ylabel('gain (sp/s)');

%% Network parameters
Net = [1.1,-.15*1;...
      -.15*1,1.1]*60;
s1 = Net(1,1);  s2 = Net(2,2); c12 = Net(1,2); c21 = Net(2,1);

%% Set up multi-neuron GLM

nneur = 2;
k = [gg.k -gg.k];%[.9 .9]; % stimulus weights
gg.k = permute(k,[1,3,2]);  % stimulus weights

gg.iht = iht;
gg.ih = zeros(nht,nneur,nneur);
gg.ih(:,:,1) = [s1*ihself, c12*ihcpl]; % input weights to neuron 1
gg.ih(:,:,2) = [c21*ihcpl, s2*ihself]; % input weights to neuron 2

%% ===== Generate some training data =============================== %%
swid = 1; % width of stimulus
stimsd = 1;  % contrast of stimulus
slen = 1e4;  % Stimulus length (in bins);
moav = 130;  % smoothing window
S1 = conv(stimsd*randn(slen,swid),ones(moav,1),'same') + 30*sin([1:slen]/400)';  %stimsd*randn(slen,swid); % Gaussian white noise stimulus
S2 = conv(stimsd*randn(slen,swid),ones(130,1),'same') + 0*sin([1:slen]/400)';
Stim = [S1, S2];
%Stim = zscore(Stim);
%Stim(Stim>0) = 1;  Stim(Stim<=0) = 0;
[tsp,sps,Itot,ispk] = simGLM_channel(gg,Stim);  % run model
tt = (dt:dt:slen*dt)';

figure()
plot(tt,sps,'-o')
ylabel('spikes','Fontsize',15)
xlabel('time (s)','Fontsize',15)
set(gca,'FontSize',15) 
figure()
imagesc([0,max(tt)],[1,2],sps')
ylabel('cell','Fontsize',15)
xlabel('time (s)','Fontsize',15)
set(gca,'FontSize',15) 

%%  ===== Fitting =============================== %%
% Initialize param struct for fitting 
kbasis = makeBasis_StimKernel(gg.ktbasprs,nkt);

gg1in = makeFittingStruct_GLM(dt,dt, nkt, size(kbasis,2), gg.k(:,1,1));  % Initialize params for fitting struct 
% gg1in = makeFittingStruct_GLM(dt,dt);  %%for single weight kernel
% gg1in.kt = gg.kt; %1.; % initial params from scaled-down sta

% Initialize fields (using h bases computed above)
gg1in.ktbas = kbasis; % k basis
gg1in.ihbas = ihbas; % h self-coupling basis
gg1in.ihbas2 = ihbas2; % h coupling-filter basis 
gg1in.ihw = wself; %randn(nhbasis,1); % init params for self-coupling filter
gg1in.ihw2 = wcpl; %randn(nhbasis2,nneur-1); % init params for cross-coupling filter
gg1in.ih = [gg1in.ihbas*gg1in.ihw*s1 gg1in.ihbas2*gg1in.ihw2*c12];%gg.ih;%
gg1in.iht = iht;
gg1in.dc = gg.dc; % Initialize dc term to zero

kt = [gg1in.kt -gg1in.kt];%[.9 .9]; % stimulus weights
% Set fields for fitting cell #1
cellnum = 1;
gg1in.k = squeeze(gg.k(:,:,cellnum));  % initial setting of k filter
gg1in.kt = gg1in.kt; %permute(kt(:,cellnum),[1,3,2]);
% gg1in.k = 1;
% gg1in.kt = 1;
couplednums = [2];  % the cells coupled to this one
gg1in.couplednums = couplednums; % cell numbers of cells coupled to this one 
gg1in.sps = sps(:,cellnum);  % Set spike responses for cell 1 
gg1in.sps2 = sps(:,couplednums); % spikes from coupled cells

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set fields for fitting cell #1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cellnum = 2;
couplednums = 1;
gg2in = gg1in;
% gg2in = makeFittingStruct_GLM(dt,dt, nkt, size(kbasis,2), -gg.k(:,1,1));  % Initialize params for fitting struct  
% gg1in = makeFittingStruct_GLM(dt,dt);  %%for single weight kernel
% gg1in.kt = gg.kt; %1.; % initial params from scaled-down sta

% Initialize fields (using h bases computed above)
% gg2in.ktbas = kbasis; % k basis
gg2in.ihbas = ihbas; % h self-coupling basis
gg2in.ihbas2 = ihbas2; % h coupling-filter basis 
gg2in.ihw = wself; %randn(nhbasis,1); % init params for self-coupling filter
gg2in.ihw2 = wcpl; %randn(nhbasis2,nneur-1); % init params for cross-coupling filter
% gg2in.ih = [gg2in.ihbas*gg1in.ihw*s2 gg1in.ihbas2*gg2in.ihw2*c21];%gg.ih;%%gg.ih;%
% gg2in.iht = iht;
% gg2in.dc = gg.dc; % Initialize dc term to zero
% gg2in.k = squeeze(gg.k(:,:,cellnum));  % initial setting of k filter
% gg2in.kt = permute(kt(:,cellnum),[1,3,2]);
gg2in.ih = [gg2in.ihbas*gg2in.ihw*s2 gg2in.ihbas2*gg2in.ihw2*c21];%gg.ih;%
gg2in.k = -gg1in.k;
gg2in.sps = sps(:,cellnum); % cell 2 spikes
gg2in.sps2 = sps(:,couplednums); % spike trains from coupled cells 
gg2in.couplednums = couplednums; % cells coupled to this one


% --- Create design matrix extract initial params from gg ----------------
% [prs_,Xstruct] = setupfitting_GLM(gg1in,Stim);
% [neglogli,~,H] = Loss_GLM_logli(prs_,Xstruct);
% -neglogli

%% Test for Bayesian decoding
figure;
lamb = Itot(:,1);
yy = dt*lamb.*exp(dt*-lamb);
xx = Stim(:,1);
factor = lsqr(yy,xx);
plot(xx)
hold on
plot(yy/dt)

corrcoef(xx,yy)
