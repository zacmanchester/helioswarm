function e = lserror(x,x_node,yhist,t_samp,t_ephem,x_ephem)

Nsat = 2;

options = odeset('RelTol',1e-8,'AbsTol',1e-8);
xswarm = [];
for k = 1:(Nsat-1)
[t,xsat] = ode113(@(t,x)dynamics_mex(t,x,t_ephem,x_ephem),t_samp,x(6*(k-1)+(1:6)),options);
xswarm = [xswarm; xsat'];
end

y = zeros(4,length(t_samp));
for k = 1:length(t_samp)
    y(:,k) = observation([x_node(:,k); xswarm(:,k)]);
end

e = vec(y(4,:)-yhist(4,:));

end

