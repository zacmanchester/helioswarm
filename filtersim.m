clear all

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

t_samp = t0:(15/24/60):(t0+14); %once every 15 minutes

Nsat = 2;
x0swarm = [eye(6); diag(ones(6,1)+[1e-4*randn(3,1); 1e-5*randn(3,1)])]*x0; %; diag(ones(6,1)+[1e-4*randn(3,1); 1e-5*randn(3,1)]); diag(ones(6,1)+[1e-4*randn(3,1); 1e-5*randn(3,1)])]*x0;

options = odeset('RelTol',1e-9,'AbsTol',1e-9);
tic
[t,xswarm] = ode113(@(t,x)swarm_dynamics(t,x,t_ephem,x_ephem),t_samp,x0swarm,options);
xswarm = xswarm';
toc

xhub = xswarm(1:6,:);
xnode = xswarm(7:12,:);
yhist = zeros(1,length(t));
for k = 1:length(t)
    yhist(:,k) = observation(xnode(:,k),xhub(:,k)) + 2e-5*randn;
end


P0 = 1e-3*eye(6);
Q = 1e-5*eye(6);
R = 1e-8;

[xhist,Phist] = iekf(x0+0.0001*randn(6,1),P0,Q,R,t,yhist,xhub,t_ephem,x_ephem);

figure(1);
subplot(3,1,1);
plot(t,1000*(xhist(1,:)-xnode(1,:)));
hold on
plot(t,2000*sqrt(squeeze(Phist(1,1,:))),'r');
plot(t,-2000*sqrt(squeeze(Phist(1,1,:))),'r');
subplot(3,1,2);
plot(t,1000*(xhist(2,:)-xnode(2,:)));
hold on
plot(t,2000*sqrt(squeeze(Phist(2,2,:))),'r');
plot(t,-2000*sqrt(squeeze(Phist(2,2,:))),'r');
ylabel('Position Error (km)');
subplot(3,1,3);
plot(t,1000*(xhist(3,:)-xnode(3,:)));
hold on
plot(t,2000*sqrt(squeeze(Phist(3,3,:))),'r');
plot(t,-2000*sqrt(squeeze(Phist(3,3,:))),'r');
xlabel('Time (days)');

figure(2)
subplot(3,1,1);
plot(t,(xhist(4,:)-xnode(4,:)));
hold on
plot(t,2*(1e6/86400)*sqrt(squeeze(Phist(4,4,:))),'r');
plot(t,-2*(1e6/86400)*sqrt(squeeze(Phist(4,4,:))),'r');
subplot(3,1,2);
plot(t,(xhist(5,:)-xnode(5,:)));
hold on
plot(t,2*(1e6/86400)*sqrt(squeeze(Phist(5,5,:))),'r');
plot(t,-2*(1e6/86400)*sqrt(squeeze(Phist(5,5,:))),'r');
ylabel('Velocity Error (m/s)');
subplot(3,1,3);
plot(t,(xhist(6,:)-xnode(6,:)));
hold on
plot(t,2*(1e6/86400)*sqrt(squeeze(Phist(6,6,:))),'r');
plot(t,-2*(1e6/86400)*sqrt(squeeze(Phist(6,6,:))),'r');
xlabel('Time (days)');

% figure();
% subplot(3,2,1);
% plot(t,1000*(xhist(7,:)-xswarm(7,:)));
% hold on
% plot(t,1000*sqrt(squeeze(Phist(7,7,:))),'r');
% plot(t,-1000*sqrt(squeeze(Phist(7,7,:))),'r');
% subplot(3,2,3);
% plot(t,1000*(xhist(8,:)-xswarm(8,:)));
% hold on
% plot(t,1000*sqrt(squeeze(Phist(8,8,:))),'r');
% plot(t,-1000*sqrt(squeeze(Phist(8,8,:))),'r');
% subplot(3,2,5);
% plot(t,1000*(xhist(9,:)-xswarm(9,:)));
% hold on
% plot(t,1000*sqrt(squeeze(Phist(9,9,:))),'r');
% plot(t,-1000*sqrt(squeeze(Phist(9,9,:))),'r');
% subplot(3,2,2);
% plot(t,(xhist(10,:)-xswarm(10,:)));
% hold on
% plot(t,sqrt(squeeze(Phist(10,10,:))),'r');
% plot(t,-sqrt(squeeze(Phist(10,10,:))),'r');
% subplot(3,2,4);
% plot(t,(xhist(5,:)-xswarm(11,:)));
% hold on
% plot(t,sqrt(squeeze(Phist(11,11,:))),'r');
% plot(t,-sqrt(squeeze(Phist(11,11,:))),'r');
% subplot(3,2,6);
% plot(t,(xhist(12,:)-xswarm(12,:)));
% hold on
% plot(t,sqrt(squeeze(Phist(12,12,:))),'r');
% plot(t,-sqrt(squeeze(Phist(12,12,:))),'r');

