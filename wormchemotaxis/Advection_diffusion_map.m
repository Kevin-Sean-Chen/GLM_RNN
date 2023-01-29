%%%Advection_diffusion_map

%%% modeling map for now
D = 0.1;  %diffusion coefficient
C0 = 110;  %initial concentration (~mM for now)
T = 500000;  %equilibrium time (seconds)
x0 = 3000;  %odor source (x0,y0)
y0 = 1250;
target = [x0, y0];
eps = 10^-15  %for numerical stability

xx = 1:3000;
yy = 1:2500;
M = zeros(length(xx), length(yy));
for ii = 1:length(xx)
    for jj = 1:length(yy)
        
        x = xx(ii);
        y = yy(jj);
        Cx = C0*(1-erf(abs(x-x0)/(sqrt(4*D*T))));   %Advection direction
        Cy = Cx/(sqrt(4*D*T)) * exp(-(y-y0)^2/(4*D*T));   %Diffusion direction
        M(ii,jj) = Cy + eps;
    end
end

figure()
imagesc(M')
% function Cxy = Adv_Dif_map(x,y,x0,y0,t)
%     Cx = C0*(1-erf(abs(x-x0)/(sqrt(4*D*t))));
%     Cy = Cx/(sqrt(4*D*t)) * exp(-(y-y0)^2/(4*D*t));
%     Cxy = Cy*1;
% end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% load test tracks
load('\\tigress-cifs.princeton.edu\fileset-leifer\Kevin\20200303_GWN_app_flow_test\Data20200303_181304\centerline_deleted_tracks.mat')
%% plotting
figure()
imagesc(log(M'))
hold on
for ii = 1:1:length(deleted_tracks)
    temp = deleted_tracks(ii).Path; 
%     plot(temp(:,1),temp(:,2),'k'); 
    hh = plot(temp(:,1),temp(:,2),'k','LineWidth',.5); hh.Color(4) = 0.9;
    hold on;
%     pause();
end
set(gca,'YDir','normal') 

%% density
figure()
tempp = [];  for ii = 1:length(deleted_tracks);  tempp = [tempp; deleted_tracks(ii).Path]; end
[nn,cc] = hist3(temp,[50,50]);
imagesc(cc{1},cc{2},nn')
set(gca,'YDir','normal')

%% sensory time series
C_xyt = {};
for ii = 1:length(deleted_tracks)
    temp = deleted_tracks(ii).Path; 
    Ct = zeros(1,size(temp,1));
    for tt = 1:size(temp,1)
        Ct(tt) = AD_map(temp(tt,1), temp(tt,2), x0, y0, D, T, C0);
    end
    C_xyt{ii} = (Ct);
end
%%
figure()
for ii = 1:length(C_xyt); plot(1/14:1/14:1/14*length(C_xyt{ii}), C_xyt{ii}-C_xyt{ii}(1)); hold on; end
%% extract dC_perp
samp_int = 10;  %down sample by integer number of points
pepr_dist = .2;  %distance perpendicular to the point
allCp = [];
allAs = [];
for ii = 1:length(deleted_tracks)
    temp = deleted_tracks(ii).Path;
    samps = 1:samp_int:length(temp);
    subs = temp(samps,:);
    angs = zeros(1,length(samps)-1);
    vecs = diff(subs);
    dCps = zeros(1,length(angs));
    
    for ss = 1:length(angs)-1
        angs(ss) = Angles(vecs(ss,:),[x0,y0]);  %(vecs(ss,:),vecs(ss+1,:));
        perp_dir = [-vecs(ss,2), vecs(ss,1)];
        perp_dir = perp_dir/norm(perp_dir);  %instantaneous perpendicular direction with unit norm
        dCps(ss) = AD_map(subs(ss,1)+perp_dir(1)*pepr_dist, subs(ss,2)+perp_dir(2)*pepr_dist, x0, y0, D, T, C0)...
                  -AD_map(subs(ss,1)-perp_dir(1)*pepr_dist, subs(ss,2)-perp_dir(2)*pepr_dist, x0, y0, D, T, C0);
    end
    
    allCp = [allCp (dCps)];
    allAs = [allAs angs];
end
%%
figure()
plot(allAs, allCp,'.');
%% angle vs. concentration
samp_int = 10;
allCs = [];
allAs = [];
for ii = 1:length(deleted_tracks)
    temp = deleted_tracks(ii).Path;
    samps = 1:samp_int:length(temp);
    subs = temp(samps,:);
    angs = zeros(1,length(samps)-1);
    vecs = diff(subs);
    
    for ss = 1:length(angs)-1
        angs(ss) = Angles(vecs(ss,:),vecs(ss+1,:));
    end
    
    con_samp = C_xyt{ii}(samps(1:end));
    plot(diff(con_samp), angs(1:end),'.'); hold on;%pause();%
    
    allCs = [allCs diff(con_samp)];
    allAs = [allAs angs];

end
%%
dl = 1;
thr = 90;
nb = 50;
tempC = allCs(1:end-dl);
tempA = allAs(dl:end);
pp = find(abs(allAs)>thr);
[a,b] = histcounts(allCs(pp),nb);
[aa,bb] = histcounts(allCs,b);
norm_ang = aa/sum(aa);
norm_pur = a/sum(a);
bins = bb(1:end-1);
bar(bins,log(norm_ang))
hold on
bar(bins,log(norm_pur))
plot(bins,norm_pur./norm_ang,'k-o')

%% speed vs. concentration
samp_int = 20;
for ii = 1:length(deleted_tracks)
    temp = deleted_tracks(ii).Path;
    samps = 1:samp_int:length(temp);
    dists = zeros(1,length(samps));
    pos_samp = temp(samps,:);
    
    for ss = 1:length(samps)-1
        dists(ss) = Distance(pos_samp(ss,:),pos_samp(ss+1,:));  %distance between points at a sampling rate
    end
    
    con_samp = C_xyt{ii}(samps);
    plot(diff(con_samp), dists(1:end-1),'.'); hold on;%pause();
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Model fitting
f = @(x)nLL_chemotaxis(x, allAs, allCp, allCs);
[x,fval] = fminunc(f,rand(1,4));%[0,5,0.1,0.1]);  %random initiation
% [x,fval] = fminunc(f,[0.5, 10, 0.5, 50]+rand(1,4));  %a closer to a reasonable value
x
fval

%% Generative model
figure;
imagesc(log(M'));
set(gca,'YDir','normal')
%dth = N(alpha*dcp,K) + (A/(1+exp(B*dC)))*U[0,2*pi];
alpha = x(1);  kappa = (1/x(2))^0.5*(180/pi);  A = x(3);  B = x(4);
%alpha = 1.1;  kappa = 0.1*(180/pi);  A = 0.4;  B = 20;
test = [];
C0 = 100000;
origin = [target(1)/2,target(2)];
target2 = target-[100,0];%origin+[500,0];%-[target(1)/2,target(2)];

allA_gen = [];
allC_gen = [];
allP_gen = [];
allths_gen = [];
for rep = 1:200
    
Tl = 1000;
dt = 1;
vm = 6.0;  %should be adjusted with the velocity statistics~~ this is approximately 0.2mm X 
vs = 1.5;
tracks = zeros(Tl,2);
tracks(1,:) = origin; %initial position
tracks(2,:) = origin+randn(1,2)*vm*dt;
ths = zeros(1,Tl);  ths(1) = randn(1); %initial angle
dxy = randn(1,2);  %change in each step

dths = zeros(1,Tl); %"recordings"
dcs = zeros(1,Tl);
dcps = zeros(1,Tl);
for t = 1+2:Tl
    dC = AD_map(tracks(t-1,1),tracks(t-1,2), x0,y0,D,T,C0) - AD_map(tracks(t-2,1),tracks(t-2,2), x0,y0,D,T,C0);
    perp_dir = [-dxy(2) dxy(1)];
    perp_dir = perp_dir/norm(perp_dir);
    dCp = AD_map(tracks(t-1,1)+perp_dir(1),tracks(t-1,2)+perp_dir(2), x0,y0,D,T,C0)...
        - AD_map(tracks(t-1,1)-perp_dir(1),tracks(t-1,2)-perp_dir(2), x0,y0,D,T,C0);
    
    wv = -alpha*dCp + kappa*randn;
    P_event = A/(1+exp(B*dC*dt));
    if rand < P_event
        beta = 1;
    else
        beta = 0;
    end
    rt = beta*(rand*360-180);
    dth = wv+rt;
    if dth>180; dth = dth-180; end;  if dth<-180; dth = dth+360; end  %within -180~180 degree range
    
    dths(t) = dth;
    dcs(t) = dC;
    dcps(t) = dCp;
    
    vv = vm+vs*randn;
    ths(t) = ths(t-1)+dth*dt;
    %if ths(t)>180; ths(t) = ths(t)-180; end;  if ths(t)<-180; ths(t) = ths(t)+360; end  %within -180~180 degree range
    e1 = [1,0];
    vec = [tracks(t-1,1)  tracks(t-1,2)]-origin; %current vector
    theta = acosd(max(-1,min((vec*e1')/norm(vec)/norm(e1),1)));  %current angle
    dd = [vv*sin(ths(t)*pi/180) vv*cos(ths(t)*pi/180)];
    R = [cos(theta*pi/180) sin(theta*pi/180); -sin(theta*pi/180) cos(theta*pi/180)];
    dxy = (R)*dd';

    tracks(t,1) = tracks(t-1,1)+dd(1)*dt;  %dxy(1)*dt;
    tracks(t,2) = tracks(t-1,2)+dd(2)*dt;  %dxy(2)*dt;
        
end

hold on
plot(tracks(:,1),tracks(:,2),'k')
plot(target(1),target(2),'ro')
plot(origin(1),origin(2),'bo')

allA_gen = [allA_gen dths];
allC_gen = [allC_gen dcs];
allP_gen = [allP_gen dCp];
allths_gen = [allths_gen ths];

end

%% Kernel version!?

%%%%%%
%% functions
function Cxy = AD_map(x,y,x0,y0,D,T,C0)
    Cx = C0*(1-erf(abs(x-x0)/(sqrt(4*D*T))));   %Advection direction
    Cxy = Cx/(sqrt(4*D*T)) * exp(-(y-y0)^2/(4*D*T));   %Diffusion direction
end

function dd = Distance(xy1,xy2)
    dd = sqrt(sum((xy1-xy2).^2));  %Distance between two point in 2D
end

function aa = Angles(v,u)%u=target  
    % with angle sign
    v_3d = [v, 0];
    u_3d = [u, 0];
    c = cross(v_3d, u_3d);
    
    % calculate degrees
    if c(3) < 0
        aa = -atan2(norm(c),dot(v,u))*(180/pi);
    else
        aa = atan2(norm(c),dot(v,u))*(180/pi);
    end
end

function dc = fold_change(u)
    posi = find(u<0);
    dc = log(abs(u));
    dc(posi) = -dc(posi);
end
