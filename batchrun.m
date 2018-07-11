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

t_samp = t0:(4/24):(t0+16); %once every 4 hours

Nsat = 2;
x0swarm = [eye(6); diag(ones(6,1)+[1e-4*randn(3,1); 1e-5*randn(3,1)])]*x0; %; diag(ones(6,1)+[1e-4*randn(3,1); 1e-5*randn(3,1)]); diag(ones(6,1)+[1e-4*randn(3,1); 1e-5*randn(3,1)])]*x0;

options = odeset('RelTol',1e-8,'AbsTol',1e-8);
xswarm = [];
for k = 1:Nsat
[t,xsat] = ode113(@(t,x)dynamics_mex(t,x,t_ephem,x_ephem),t_samp,x0swarm(6*(k-1)+(1:6)),options);
xswarm = [xswarm; xsat'];
end

xnode = xswarm(1:6,:);

yhist = zeros(Nsat+2,length(t));
for k = 1:length(t)
    yhist(:,k) = observation(xswarm(:,k));
end

lsoptions = optimoptions('lsqnonlin','Algorithm','levenberg-marquardt','Display','iter');

[x0est,res] = lsqnonlin(@(x)lserror(x,xnode,yhist,t_samp,t_ephem,x_ephem),x0 + 0*(x0swarm(7:12)-x0),[],[],lsoptions);

options = odeset('RelTol',1e-8,'AbsTol',1e-8);
xest = [];
for k = 1:(Nsat-1)
[t,xsat] = ode113(@(t,x)dynamics_mex(t,x,t_ephem,x_ephem),t_samp,x0est(6*(k-1)+(1:6)),options);
xest = [xest; xsat'];
end

figure();
plot(t,1000*vecnorm(xswarm(7:9,:)-xswarm(1:3,:)));
hold on;
plot(t,1000*vecnorm(xest(1:3,:)-xswarm(1:3,:)));
title('Range');

figure();
subplot(3,1,1);
plot(t,1000*(xswarm(7,:)-xswarm(1,:)));
hold on;
plot(t,1000*(xest(1,:)-xswarm(1,:)));
title('Relative Position');
legend('True','Estimated');
subplot(3,1,2);
plot(t,1000*(xswarm(8,:)-xswarm(2,:)));
hold on;
plot(t,1000*(xest(2,:)-xswarm(2,:)));
subplot(3,1,3);
plot(t,1000*(xswarm(9,:)-xswarm(3,:)));
hold on
plot(t,1000*(xest(3,:)-xswarm(3,:)));

