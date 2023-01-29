%Odor_map_construct
%% load data
load('odor_map.mat')
V = timeaverage{600};
[X,Y] = meshgrid(1:1:size(V,2),1:1:size(V,1));
interp = 0.25;
[Xq,Yq] = meshgrid(1:interp:size(V,2),1:interp:size(V,1));

Vq = interp2(X,Y,V,Xq,Yq,'spline');

figure
surf(Xq,Yq,Vq);
title('interpolation of the odor landscape');
xlabel('x')
ylabel('y')
zlabel('ppm')

%% accounting for sensor position
X_osa = zeros(8,14);  %correct X-Y indexing
Y_osa = zeros(8,14);

for ii = 1:8
    for jj = 1:14
        if mod(jj,2)==0
            X_osa(ii,jj) = (ii-1)*2+1;
        else
            X_osa(ii,jj) = (ii)*2;
        end
        Y_osa(ii,jj) = jj;
    end
end

%% 
sorted_osa = [];
for ii = 1:2:14
    temp_id = X_osa(:,ii:ii+1);
    temp_va = V(:,ii:ii+1);
    [aa,bb] = sort(temp_id(:));
    sorted_osa = [sorted_osa temp_va(bb)];
end
figure
imagesc(sorted_osa)

%%
[X,Y] = meshgrid(1:1:size(sorted_osa,2),1:1:size(sorted_osa,1));
interp = 0.25;
[Xq,Yq] = meshgrid(1:interp:size(sorted_osa,2),1:interp:size(sorted_osa,1));

Vq = interp2(X,Y,sorted_osa,Xq,Yq,'spline');

figure
surf(Xq,Yq,Vq);
title('interpolation of the odor landscape');
xlabel('x')
ylabel('y')
zlabel('ppm')

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% convection diffusion
D = 1;  %diffusion coefficient (cm^2/s)
V_o = 1;  %~~cm/s (?) of odor flow
V_a = 4;  %~cm/s for boundary air flows
C0 = 100;  %mM (?)
k = 10;  %outlet

T = 500;  %seconds
dt = 0.01;
time = 0:dt:T;
nt = length(time);
dy = .5;  %cm length
yy = -10:dy:10;
ny = length(yy);
dx = .5;
xx = 0:dx:15;
nx = length(xx);
C = zeros(nx,ny,nt);
mid = round(ny/2);

odor_pos = [mid-2:mid+2];
C(1,odor_pos,1) = C0;

figure;
for ti = 1:nt-1
    
    %%%Convection-Diffusion dynamics
    for xi = 2:nx-1
        for yi = 2:ny-1
            if length(find(odor_pos==yi))>0  %is in the odor flow zone
            C(xi,yi,ti+1) = C(xi,yi,ti) + D*dt/(dx*dx)*(C(xi+1,yi,ti)-2*C(xi,yi,ti)+C(xi-1,yi,ti))...
                + D*dt/(dy*dy)*(C(xi,yi+1,ti)-2*C(xi,yi,ti)+C(xi,yi-1,ti))...
                - V_o*dt/(2*dx)*(C(xi+1,yi,ti)-C(xi-1,yi,ti))...
                - 0*dt/(2*dx)*(C(xi,yi+1,ti)-C(xi,yi-1,ti));   %no velocity along y (??)
            else  %is the boundary air flow with a different flow rate
            C(xi,yi,ti+1) = C(xi,yi,ti) + D*dt/(dx*dx)*(C(xi+1,yi,ti)-2*C(xi,yi,ti)+C(xi-1,yi,ti))...
                + D*dt/(dy*dy)*(C(xi,yi+1,ti)-2*C(xi,yi,ti)+C(xi,yi-1,ti))...
                - V_a*dt/(2*dx)*(C(xi+1,yi,ti)-C(xi-1,yi,ti))...
                - 0*dt/(2*dx)*(C(xi,yi+1,ti)-C(xi,yi-1,ti));   %no velocity along y (??)                
            end
        end
    end
    
    %%%Boundary conditions
    C(1,:,ti) = 0;  %clean air
    C(1,[mid-2:mid+2],ti+1) = C0;  %source
    Q = sum(C(1,:,ti+1));
    C(nx,:,ti) = 0;%C(nx-1,:,ti)-k*C(nx-1,:,ti)*dx;  %outlet
    C(:,1,ti) = 0;%C(:,2,ti)-k*C(:,2,ti)*dy;  %ideal boundary
    C(:,ny,ti) = 0;%C(:,ny-1,ti)-k*C(:,ny-1,ti)*dy;
    
    %imagesc(squeeze(C(:,:,ti))');  pause();
    
end

%%%sptaiol section
figure;
plot(squeeze(C(:,:,ti)'))
figure;
imagesc(xx,yy,squeeze(C(:,:,end))')
title(['odor flow=',num2str(V_o),', air flow=',num2str(V_a)])


%%
%% Advection-diffuction with reversible reaction (surface absorsion)
%%%flow
D = 1;  %diffusion coefficient (cm^2/s)
V_o = 1;  %~~cm/s (?) of odor flow
V_a = 4;  %~cm/s for boundary air flows
C0 = 100;  %mM (?)
k = 10;  %outlet
%%%surface
lamb = .5;  % absorbsion
mu = .1;  %emission
w = .1;  %unit conversion
%%%setup
T = 800;  %seconds
dt = 0.01;
time = 0:dt:T;
nt = length(time);
dy = .5;  %cm length
yy = -10:dy:10;
ny = length(yy);
dx = .5;
xx = 0:dx:15;
nx = length(xx);
C = zeros(nx,ny,nt);  %concentration array
S = zeros(nx,ny,nt);  %saturation array
mid = round(ny/2);

odor_pos = [mid-2:mid+2];
C(1,odor_pos,1) = C0;

T_off = 30000;  %time of turing off odor source

% figure;
for ti = 1:nt-1
    
    %%%Convection-Diffusion dynamics
    for xi = 2:nx-1
        for yi = 2:ny-1
            if length(find(odor_pos==yi))>0  %is in the odor flow zone
%                 DS = (lamb*C(xi,yi,ti) - mu*S(xi,yi,ti));
                DS = (lamb*C(xi,yi,ti)*(1-S(xi,yi,ti)) - mu*S(xi,yi,ti));
                S(xi,yi,ti+1) = S(xi,yi,ti) + dt*DS;  %reversible interactions
                DS = (lamb*C(xi,yi,ti) - mu*S(xi,yi,ti));
                C(xi,yi,ti+1) = C(xi,yi,ti) + D*dt/(dx*dx)*(C(xi+1,yi,ti)-2*C(xi,yi,ti)+C(xi-1,yi,ti))...
                    + D*dt/(dy*dy)*(C(xi,yi+1,ti)-2*C(xi,yi,ti)+C(xi,yi-1,ti))...
                    - V_o*dt/(2*dx)*(C(xi+1,yi,ti)-C(xi-1,yi,ti))...
                    - 0*dt/(2*dx)*(C(xi,yi+1,ti)-C(xi,yi-1,ti)) - w*DS;   %no velocity along y (??)
            
            else  %is the boundary air flow with a different flow rate
%                 DS = (lamb*C(xi,yi,ti) - mu*S(xi,yi,ti));
                DS = (lamb*C(xi,yi,ti)*(1-S(xi,yi,ti)) - mu*S(xi,yi,ti));
                S(xi,yi,ti+1) = S(xi,yi,ti) + dt*DS;  %reversible interactions
                C(xi,yi,ti+1) = C(xi,yi,ti) + D*dt/(dx*dx)*(C(xi+1,yi,ti)-2*C(xi,yi,ti)+C(xi-1,yi,ti))...
                    + D*dt/(dy*dy)*(C(xi,yi+1,ti)-2*C(xi,yi,ti)+C(xi,yi-1,ti))...
                    - V_a*dt/(2*dx)*(C(xi+1,yi,ti)-C(xi-1,yi,ti))...
                    - 0*dt/(2*dx)*(C(xi,yi+1,ti)-C(xi,yi-1,ti)) - w*DS;   %no velocity along y (??)                
            
            end
        end
    end
    
    %%%Boundary conditions
    C(1,:,ti) = 0;  %clean air
    
    Q = sum(C(1,:,ti+1));
    C(nx,:,ti) = 0;%C(nx-1,:,ti)-k*C(nx-1,:,ti)*dx;  %outlet
    C(:,1,ti) = 0;%C(:,2,ti)-k*C(:,2,ti)*dy;  %ideal boundary
    C(:,ny,ti) = 0;%C(:,ny-1,ti)-k*C(:,ny-1,ti)*dy;
    if ti<T_off
        C(1,[mid-2:mid+2],ti+1) = C0;  %source
    elseif ti>=T_off
        C(1,[mid-2:mid+2],ti+1) = 0;  %off
    end
    %imagesc(squeeze(C(:,:,ti))');  pause();
    
end

%%%sptaiol section
figure;
plot(squeeze(C(:,:,ti)'))
figure;
imagesc(xx,yy,squeeze(C(:,:,end))')
title(['odor flow=',num2str(V_o),', air flow=',num2str(V_a)])

%% temporal analysis


