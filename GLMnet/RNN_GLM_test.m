%RNN_GLM_test

clear
clc
% cd ..; setpaths_GLMspiketools;
set(0,'DefaultLineLineWidth',1,...
    'DefaultLineMarkerSize',8, ...
    'DefaultAxesLineWidth',2, ...
    'DefaultAxesFontSize',14,...
    'DefaultAxesFontWeight','Bold');
%% network settings

Nunits = 2;             % Number of units to simulate

attractor_flag = 1;         % Allowed values are 1, 2, 3, 5, 6
% attractor_flag determines the connection strengths and the type of
% dynamical system produced.
% 1: point attractor                            Figure 1 in the paper
% 2: multistable point attractor system         Figure 2 in the paper
% 3: inhibition stabilized point attractor      Figure 3 in the paper
% 4: should not be used with two units
% 5: marginal state (line attractor)            Figure 5 in the paper
% 6: oscillator (bistable)                      Figure 6 in the paper
% 7: should not be used with two units
% 8: should not be used with two units
% 9: should not be used with two units
tmax = 3;  % default value of maximum time to simulate

switch attractor_flag        
    case 1,
        % case 2 is a multistable system with three point attractor states
        % (all cells at low rates or cell 1 high and cell 3 low or cell 3
        % high and cell 1 low. In such a system some cells can produce more
        % graded activity (cell 2) due to changes in input from the others.
        % Initial conditions or the types of input pulse determine the
        % final attractor state, so the system exhibits memory.
        Ithresh = [5; 5];
        W = [1.1 -0.15; -0.15 1.1];
        rinit1 = [50; 55];
        Iapp1 = [ 0; 30];
        Iapp2 = [30; 0];
        
    case 2,
        % case 5 produces a marginal state, or line attractor as the line
        % described by r1 + r2 = 200/3 is a continuous set of fixed points
        % in the range 0 < r1,r2 < rmax (= 100).
        % Final set of activities depends on the initial values and current
        % pulses are integrated.
        Ithresh = [-20; -20];
        W = [0.7 -0.3; -0.3 0.7];
        rinit1 = [30; 75];
        Iapp1 = [1; 0];
        Iapp2 = [2; 0];
        
    case 3,
        % case 6 produces an oscillator with a triphasic rhythm. Changes in
        % initial conditions alter the phase of the oscillation but not the
        % pattern, which is an orbit attractor or limit cycle.
        Ithresh = [8; 20];
        W = [2.2 -1.3; 1.2 -0.1];
        rinit1 = [80; 0];
        Iapp1 = [0; 0];
        Iapp2 = [-10; 0];
        
    otherwise
        disp('system undefined')
end

%% Set up the time vector
dt = 0.001;
tvec = 0:dt:tmax;
Nt = length(tvec);

r = zeros(Nunits,Nt);   % array of rate of each cell as a function of time
rmax = 100;             % maximum firing rate
tau = 0.010;            % base time constant for changes of rate

%% Set up details of an applied current used in some systems
Iapp = zeros(Nunits,Nt);    % Array of time-dependent and unit-dependent current
Ion1 = 1;                    % Time to switch on
Ion2 = 2;                    % Time to switch on
Idur = 0.2;                 % Duration of current

non1 = round(Ion1/dt);            % Time step to switch on current
noff1 = round((Ion1+Idur)/dt);    % Time step to switch off current
non2 = round(Ion2/dt);            % Time step to switch on current
noff2 = round((Ion2+Idur)/dt);    % Time step to switch off current

% Add the applied current pulses
Iapp(:,non1:noff1) = Iapp1*ones(1,noff1-non1+1);
Iapp(:,non2:noff2) = Iapp2*ones(1,noff2-non2+1);

r(:,1) = rinit1;                % Initialize firing rate

%% spiking parameters
tau_m = 0.01;
tau_r = 0.05;
tau_d = 0.01;
switch attractor_flag
    case 1,
        v_the = 13;
        v_res = -10;
        lamb = 160;
    case 2,
        v_the = 7.5;
        v_res = -2;
        lamb = 75;
    case 3,
        v_the = 9.5;
        v_res = -1;
        lamb = 35;
end

%%
vm = zeros(Nunits,Nt);
rs = zeros(Nunits,Nt);
ss = zeros(Nunits,Nt);
spk = zeros(Nunits,Nt);

%% spiking RNN dynamics
for tt = 2:Nt
    %dynamics
    vm(:,tt) = vm(:,tt-1) + dt*(1/tau_m)*(-vm(:,tt-1) + lamb* W*rs(:,tt-1) + Iapp(:,tt-1)*10+10);  %membrane potential  *10+10
    rs(:,tt) = rs(:,tt-1) + dt*(-rs(:,tt-1)/tau_d + ss(:,tt-1));  %spike rate
    ss(:,tt) = ss(:,tt-1) + dt*(-ss(:,tt-1)/tau_r + 1/(tau_d*tau_r)*spk(:,tt-1));  %synaptic input
    %spiking process
    poss = find(vm(:,tt)>v_the);  %recording spikes
    if ~isempty(poss)
        spk(poss,tt) = 1;
    end
    posr = find(spk(:,tt-1)>0);  %if spiked
    if ~isempty(posr)
        vm(posr,tt) = v_res;  %reseting spiked neurons
        spk(posr,tt) = 0;
    end
%     I = W*r(:,i-1) + Iapp(:,i-1);                   % total current to each unit
%     newr = r(:,i-1) + dt/tau*(I-Ithresh-r(:,i-1));  % Euler-Mayamara update of rates
%     r(:,i) = max(newr,0);                           % rates are not negative
%     r(:,i) = min(r(:,i),rmax);                      % rates can not be above rmax
  
end
%% plotting
figure()
subplot(3,1,1)
plot(tvec,vm)
ylabel('potential','Fontsize',15)
set(gca,'FontSize',15) 
subplot(3,1,2)
% plot(tvec,spk,'-o')
imagesc([0,max(tvec)],[1 2],spk)
ylabel('cell','Fontsize',15)
set(gca,'FontSize',15) 
subplot(3,1,3)
plot(tvec,Iapp,'LineWidth',2)
xlabel('time (s)','Fontsize',15)
ylabel('stimuli','Fontsize',15)
set(gca,'FontSize',15) 

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% GLM inference %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
%Initialize kernel methods
% Make basis for self-coupling term
nb = 7;
ihbasprs.ncols = nb; % number of basis vectors
ihbasprs.hpeaks = [.001, .1/3]; % peak of 1st and last basis vector
ihbasprs.b = .01;  % scaling (smaller -> more logarithmic scalin
ihbasprs.absref = .001; % absolute refractory period basis vector (optional)
% Make basis 
[iht,ihbas,ihbasis] = makeBasis_PostSpike(ihbasprs,dt);
nht = length(iht); % number of bins

% Make basis for cross-coupling term
ihbasprs2.ncols = nb;  % number of basis vectors
ihbasprs2.hpeaks = [0.001,.1/3]; % put peak at 10ms and "effective" 1st peak at 0
ihbasprs2.b = .01;  % smaller -> more logarithmic scaling
ihbasprs2.absref = .001; % no abs-refracotry period for this one
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


%%
% Initialize param struct for fitting 
gg1in = makeFittingStruct_GLM(dt,dt);  % Initialize params for fitting struct 

% Initialize fields (using h bases computed above)
gg1in.ktbas = 1; % k basis
gg1in.ihbas = ihbas; % h self-coupling basis
gg1in.ihbas2 = ihbas2; % h coupling-filter basis
nktbasis = 1; % number of basis vectors in k basis
nhbasis = size(ihbas,2); % number of basis vectors in h basis
nhbasis2 = size(ihbas2,2); % number of basis vectors in h basis
gg1in.kt = 1; % initial params from scaled-down sta 
gg1in.k = 1;  % initial setting of k filter
gg1in.ihw = zeros(nhbasis,1); % init params for self-coupling filter
gg1in.ihw2 = zeros(nhbasis2,Nunits-1); % init params for cross-coupling filter
gg1in.ih = [gg1in.ihbas*gg1in.ihw gg1in.ihbas2*gg1in.ihw2];
gg1in.iht = iht;
gg1in.dc = 0; % Initialize dc term to zero

%%
% Set fields for fitting cell #1
couplednums = [2];  % the cells coupled to this one
gg1in.couplednums = couplednums; % cell numbers of cells coupled to this one 
gg1in.sps = spk(1,:)';  % Set spike responses for cell 1 
gg1in.sps2 = spk(couplednums,:)'; % spikes from coupled cells

% Compute initial value of negative log-likelihood (just to inspect)
Stim = Iapp(1,:)';% + randn(Nt,1);
[neglogli0,rr] = neglogli_GLM(gg1in,Stim);

% Do ML fitting
fprintf('Fitting neuron 1:  initial neglogli0 = %.3f\n', neglogli0);
opts = {'display', 'iter', 'maxiter', 100};
[gg1, neglogli1] = MLfit_GLM(gg1in,Stim,opts); % do ML (requires optimization toolbox)


cellnum = 2;
couplednums = [1];%setdiff(1:2, cellnum); % the cells coupled to this one

gg2in = gg1in; % initial parameters for fitting 
gg2in.sps = spk(cellnum,:)'; % cell 2 spikes
gg2in.sps2 = spk(couplednums,:)'; % spike trains from coupled cells 
gg2in.couplednums = couplednums; % cells coupled to this one

% Compute initial value of negative log-likelihood (just to inspect)
Stim = Iapp(2,:)';% + randn(Nt,1);
[neglogli0,rr] = neglogli_GLM(gg1in,Stim);

% Do ML fitting
fprintf('Fitting neuron 2\n');
[gg2, neglogli2] = MLfit_GLM(gg2in,Stim,opts); % do ML (requires optimization toolbox)

%%
%plotting
figure()
colors = get(gca,'colororder');
set(gcf,'defaultAxesColorOrder',colors(1:3,:)); % use only 3 colors
lw = 2; % linewidth
ymax = max(exp([gg1.ih(:)])); % max of y range

subplot(121)
plot(gg1.iht, exp((gg1.ih)), '--', 'linewidth', lw);
%legend('true h11', 'true h21', 'true h31', 'estim h11', 'estim h21', 'estim h31', 'location', 'southeast');
title('incoming filters: cell 1'); axis tight; set(gca,'ylim',[0,ymax]); 
ylabel('gain (sp/s)'); xlabel('time after spike (s)');
set(gca,'FontSize',15) 
subplot(122)
plot(gg2.iht, exp((gg2.ih)), '--', 'linewidth', lw);
title('incoming filters: cell 2'); axis tight; set(gca,'ylim',[0,ymax]); 
ylabel('gain (sp/s)'); xlabel('time after spike (s)');
set(gca,'FontSize',15) 

%%
%% ===== Generate some training data =============================== %%
gg = makeSimStruct_GLM(1,dt,dt);
k = [.1 .1]*10; % stimulus weights
gg.k = permute(k,[1,3,2]);  % stimulus weights

gg.iht = iht;
gg.ih = zeros(nht,Nunits,Nunits);
gg.ih(:,:,1) = [gg1.ih]; % input weights to neuron 1
gg.ih(:,:,2) = [fliplr(gg2.ih)]; % input weights to neuron 2
[tsp,sps,Itot,ispk] = simGLM_channel(gg,Iapp');  % run model

figure()
subplot(211)
plot(tvec,sps)
title('GLM generative')
ylabel('potential','Fontsize',15)
set(gca,'FontSize',15) 
subplot(212)
imagesc([0,max(tvec)],[1,2],sps')
ylabel('cell','Fontsize',15)
xlabel('time (s)','Fontsize',15)
set(gca,'FontSize',15) 
