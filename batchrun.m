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

dt = 4/24; %once every 4 hours
t_samp = t0:dt:(t0+16);

x0node = diag(ones(6,1)+[1e-4*randn(3,1); 1e-5*randn(3,1)])*x0hub;

options = odeset('RelTol',1e-8,'AbsTol',1e-8);
[t,xhub] = ode113(@(t,x)dynamics_mex(t,x,t_ephem,x_ephem),t_samp,x0hub,options);
xhub = xhub';
[~,xnode] = ode113(@(t,x)dynamics_mex(t,x,t_ephem,x_ephem),t_samp,x0node,options);
xnode = xnode';

yhist = zeros(1,length(t));
for k = 1:length(t)
    yhist(:,k) = observation(xnode(:,k),xhub(:,k)) + 1e-5*randn;
end

lsoptions = optimoptions('lsqnonlin','Algorithm','levenberg-marquardt','Display','iter');
[x0est,res] = lsqnonlin(@(x)lserror(x,xhub,yhist,t_samp,t_ephem,x_ephem),x0hub + 1e-3*(x0node-x0hub),[],[],lsoptions);

[~,xest] = ode113(@(t,x)dynamics_mex(t,x,t_ephem,x_ephem),t_samp,x0est,options);
xest = xest';

figure();
plot(t,1000*vecnorm(xnode(1:3,:)-xhub(1:3,:)));
hold on;
plot(t,1000*vecnorm(xest(1:3,:)-xhub(1:3,:)));
title('Range');

figure();
subplot(3,1,1);
plot(t,1000*(xnode(1,:)-xhub(1,:)));
hold on;
plot(t,1000*(xest(1,:)-xhub(1,:)));
title('Relative Position');
legend('True','Estimated');
subplot(3,1,2);
plot(t,1000*(xnode(2,:)-xhub(2,:)));
hold on;
plot(t,1000*(xest(2,:)-xhub(2,:)));
subplot(3,1,3);
plot(t,1000*(xnode(3,:)-xhub(3,:)));
hold on
plot(t,1000*(xest(3,:)-xhub(3,:)));

figure();
subplot(3,1,1);
plot(t,1000*(xest(1,:)-xnode(1,:)));
subplot(3,1,2);
plot(t,1000*(xest(2,:)-xnode(2,:)));
ylabel('Position Error (km)');
subplot(3,1,3);
plot(t,1000*(xest(3,:)-xnode(3,:)));
xlabel('Time (days)');
