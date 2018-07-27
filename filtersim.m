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
x0hub = [r0/1000; v0*(24*60*60/1000)];
t0 = 25 + 22/24 + 30/(24*60) + 25.52/(24*60*60);

dt = 1/24; %once every 60 min
t_samp = t0:dt:(t0+16);

x0node = diag(ones(6,1)+[1e-4*randn(3,1); 1e-5*randn(3,1)])*x0hub;

options = odeset('RelTol',1e-8,'AbsTol',1e-8);
[t,xhub] = ode113(@(t,x)dynamics_mex(t,x,t_ephem,x_ephem),t_samp,x0hub,options);
xhub = xhub';
[~,xnode] = ode113(@(t,x)dynamics_mex(t,x,t_ephem,x_ephem),t_samp,x0node,options);
xnode = xnode';

% [t,xhub] = ode4(@(t,x)dynamics_mex(t,x,t_ephem,x_ephem),[t_samp(1) t_samp(end)],x0hub,dt);
% [t,xnode] = ode4(@(t,x)dynamics_mex(t,x,t_ephem,x_ephem),[t_samp(1) t_samp(end)],x0node,dt);

yhist = zeros(1,length(t));
for k = 1:length(t)
    yhist(:,k) = observation(xnode(:,k),xhub(:,k)) + 1e-5*randn;
end

U0 = .1*eye(6);
V = blkdiag(1e-4*eye(3), 1e-2*eye(3));
W = 1e-5;

[xhist1,Uhist1] = isrekf(x0hub + 1e-3*(x0node-x0hub),U0,V,W,t,yhist,xhub,t_ephem,x_ephem);

lsoptions = optimoptions('lsqnonlin','Algorithm','levenberg-marquardt','Display','iter');
[x0est,res] = lsqnonlin(@(x)lserror(x,xhub,yhist,t_samp,t_ephem,x_ephem),x0hub + 1e-3*(x0node-x0hub),[],[],lsoptions);

[~,xhist3] = ode4(@(t,x)dynamics_mex(t,x,t_ephem,x_ephem),[t_samp(1) t_samp(end)],x0est,dt);

Phist1 = zeros(size(Uhist1));
for k = 1:length(t)
    Phist1(:,:,k) = Uhist1(:,:,k)'*Uhist1(:,:,k);
end

% Phist2 = zeros(size(Uhist2));
% for k = 1:length(t)
%     Phist2(:,:,k) = Uhist2(:,:,k)'*Uhist2(:,:,k);
% end

figure(1);
subplot(3,1,1);
plot(t,1000*(xhist1(1,:)-xnode(1,:)));
hold on
plot(t,2000*sqrt(squeeze(Phist1(1,1,:))),'r');
plot(t,-2000*sqrt(squeeze(Phist1(1,1,:))),'r');
subplot(3,1,2);
plot(t,1000*(xhist1(2,:)-xnode(2,:)));
hold on
plot(t,2000*sqrt(squeeze(Phist1(2,2,:))),'r');
plot(t,-2000*sqrt(squeeze(Phist1(2,2,:))),'r');
ylabel('Position Error (km)');
subplot(3,1,3);
plot(t,1000*(xhist1(3,:)-xnode(3,:)));
hold on
plot(t,2000*sqrt(squeeze(Phist1(3,3,:))),'r');
plot(t,-2000*sqrt(squeeze(Phist1(3,3,:))),'r');
xlabel('Time (days)');

% figure(2);
% subplot(3,1,1);
% plot(t,1000*(xhist2(1,:)-xnode(1,:)));
% hold on
% plot(t,2000*sqrt(squeeze(Phist2(1,1,:))),'r');
% plot(t,-2000*sqrt(squeeze(Phist2(1,1,:))),'r');
% subplot(3,1,2);
% plot(t,1000*(xhist2(2,:)-xnode(2,:)));
% hold on
% plot(t,2000*sqrt(squeeze(Phist2(2,2,:))),'r');
% plot(t,-2000*sqrt(squeeze(Phist2(2,2,:))),'r');
% ylabel('Position Error (km)');
% subplot(3,1,3);
% plot(t,1000*(xhist2(3,:)-xnode(3,:)));
% hold on
% plot(t,2000*sqrt(squeeze(Phist2(3,3,:))),'r');
% plot(t,-2000*sqrt(squeeze(Phist2(3,3,:))),'r');
% xlabel('Time (days)');

figure(3);
subplot(3,1,1);
plot(t,1000*(xhist3(1,:)-xnode(1,:)));
hold on
plot(t,1000*(xhist1(1,:)-xnode(1,:)));
legend('batch','EKF');
subplot(3,1,2);
plot(t,1000*(xhist3(2,:)-xnode(2,:)));
hold on
plot(t,1000*(xhist1(2,:)-xnode(2,:)));
ylabel('Position Error (km)');
subplot(3,1,3);
plot(t,1000*(xhist3(3,:)-xnode(3,:)));
hold on
plot(t,1000*(xhist1(3,:)-xnode(3,:)));
xlabel('Time (days)');

% figure(4);
% subplot(3,1,1);
% plot(t,1000*(xnode(1,:)-xhub(1,:)));
% hold on;
% plot(t,1000*(xhist3(1,:)-xhub(1,:)));
% title('Relative Position');
% legend('True','Estimated');
% subplot(3,1,2);
% plot(t,1000*(xnode(2,:)-xhub(2,:)));
% hold on;
% plot(t,1000*(xhist3(2,:)-xhub(2,:)));
% subplot(3,1,3);
% plot(t,1000*(xnode(3,:)-xhub(3,:)));
% hold on
% plot(t,1000*(xhist3(3,:)-xhub(3,:)));

% figure(3)
% subplot(3,1,1);
% plot(t,(xhist1(4,:)-xnode(4,:)));
% hold on
% plot(t,2*(1e6/86400)*sqrt(squeeze(Phist1(4,4,:))),'r');
% plot(t,-2*(1e6/86400)*sqrt(squeeze(Phist1(4,4,:))),'r');
% subplot(3,1,2);
% plot(t,(xhist1(5,:)-xnode(5,:)));
% hold on
% plot(t,2*(1e6/86400)*sqrt(squeeze(Phist1(5,5,:))),'r');
% plot(t,-2*(1e6/86400)*sqrt(squeeze(Phist1(5,5,:))),'r');
% ylabel('Velocity Error (m/s)');
% subplot(3,1,3);
% plot(t,(xhist1(6,:)-xnode(6,:)));
% hold on
% plot(t,2*(1e6/86400)*sqrt(squeeze(Phist1(6,6,:))),'r');
% plot(t,-2*(1e6/86400)*sqrt(squeeze(Phist1(6,6,:))),'r');
% xlabel('Time (days)');
% 
% figure(4)
% subplot(3,1,1);
% plot(t,(xhist2(4,:)-xnode(4,:)));
% hold on
% plot(t,2*(1e6/86400)*sqrt(squeeze(Phist2(4,4,:))),'r');
% plot(t,-2*(1e6/86400)*sqrt(squeeze(Phist2(4,4,:))),'r');
% subplot(3,1,2);
% plot(t,(xhist2(5,:)-xnode(5,:)));
% hold on
% plot(t,2*(1e6/86400)*sqrt(squeeze(Phist2(5,5,:))),'r');
% plot(t,-2*(1e6/86400)*sqrt(squeeze(Phist2(5,5,:))),'r');
% ylabel('Velocity Error (m/s)');
% subplot(3,1,3);
% plot(t,(xhist2(6,:)-xnode(6,:)));
% hold on
% plot(t,2*(1e6/86400)*sqrt(squeeze(Phist2(6,6,:))),'r');
% plot(t,-2*(1e6/86400)*sqrt(squeeze(Phist2(6,6,:))),'r');
% xlabel('Time (days)');

figure(5)
plot(t,1000*vecnorm(xnode(1:3,:)-xhub(1:3,:)));
ylabel('Hub-Node Distance (km)');
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

