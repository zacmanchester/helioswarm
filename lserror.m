function e = lserror(x0,xhub,yhist,t_samp,t_ephem,x_ephem)

%[~,xnode] = ode4(@(t,x)dynamics_mex(t,x,t_ephem,x_ephem),[t_samp(1) t_samp(end)],x0,dt);

options = odeset('RelTol',1e-8,'AbsTol',1e-8); 
[~,xnode] = ode113(@(t,x)dynamics_mex(t,x,t_ephem,x_ephem),t_samp,x0,options); 
xnode = xnode';

y = zeros(1,length(t_samp));
for k = 1:length(t_samp)
    y(k) = observation(xnode(:,k),xhub(:,k));
end

e = y(:)-yhist(:);

end

