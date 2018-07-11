%Load sun + moon ephem data
[t_sun,x_sun] = readEphemData('horizons_sun.txt');
[t_moon,x_moon] = readEphemData('horizons_moon.txt');

t_ephem = t_sun;
x_ephem = [x_sun(1:3,:)/1000; x_moon(1:3,:)/1000]; %put distance in 1000*km

%Initial Conditions
% 25 Jan 2024 22:30:25.520
a = 243729.554542; %km
e = 0.598335;
i = 51.975*(pi/180);
RAAN = 61.110*(pi/180);
argp = 247.345*(pi/180);
nu = 0.0; %true anomaly
p = a*(1-e*e); %semilatus rectum

%Convert to ECI state vector
[r0,v0] = coe2rv(p,e,i,RAAN,argp,nu);

%Convert units to 1000*km and days
x0 = [r0/1000; v0*(24*60*60/1000)];
t0 = 25 + 22/24 + 30/(24*60) + 25.52/(24*60*60);

%Sanity checks
% T = 2*pi*sqrt(a^3/398600.440)/60/60/24;
% a_sun = norm(x_ephem(1:3,1));
% T_sun = 2*pi*sqrt((a_sun^3)/(1.3271244004193938e11*((60*60*24)^2/(1000^3))));

%Derivative checks
% delta = 1e-6*eye(6);
% A_fd = zeros(6);
% [xd0,A] = dynamics(t0,x0,t_ephem,x_ephem);
% for k = 1:6
%     A_fd(:,k) = (dynamics(t0,x0+delta(:,k),t_ephem,x_ephem) - dynamics(t0,x0-delta(:,k),t_ephem,x_ephem))/2e-6;
% end
% delta = 1e-6*eye(18);
% C_fd = zeros(5,18);
% x_obs = [eye(6); diag(ones(6,1)+1e-2*randn(6,1)); diag(ones(6,1)+1e-2*randn(6,1))]*x0;
% [y0,C] = observation(x_obs);
% for k = 1:18
%     C_fd(:,k) = (observation(x_obs+delta(:,k)) - observation(x_obs-delta(:,k)))/2e-6;
% end

t_samp = t0:(1/24):(t0+30);

Nsat = 4;
x0swarm = [eye(6); diag(ones(6,1)+[1e-4*randn(3,1); 1e-5*randn(3,1)]); diag(ones(6,1)+[1e-4*randn(3,1); 1e-5*randn(3,1)]); diag(ones(6,1)+[1e-4*randn(3,1); 1e-5*randn(3,1)])]*x0;

options = odeset('RelTol',1e-8,'AbsTol',1e-8);
xswarm = [];
for k = 1:Nsat
[t,xsat] = ode113(@(t,x)dynamics_mex(t,x,t_ephem,x_ephem),t_samp,x0swarm(6*(k-1)+(1:6)),options);
xswarm = [xswarm; xsat'];
end

G = obsgram(t,xswarm,@(t,x)dynamics_mex(t,x,t_ephem,x_ephem),@(x)observation(x));
sig = svd(G);
min(sig)

r_samp = ephemInterp(t_samp,t_ephem,x_ephem);

figure(1);
plot3(xswarm(1,:),xswarm(2,:),xswarm(3,:),'b');
hold on;
plot3(xswarm(1,end),xswarm(2,end),xswarm(3,end),'bo');
plot3(0,0,0,'go');
plot3(r_samp(4,:),r_samp(5,:),r_samp(6,:),'r');
plot3(r_samp(4,end),r_samp(5,end),r_samp(6,end),'ro');
hold off;

figure(2);
plot(t,1000*vecnorm(xswarm(7:9,:)-xswarm(1:3,:)));
hold on;
plot(t,1000*vecnorm(xswarm(13:15,:)-xswarm(1:3,:)));
plot(t,1000*vecnorm(xswarm(19:21,:)-xswarm(1:3,:)));
hold off;
