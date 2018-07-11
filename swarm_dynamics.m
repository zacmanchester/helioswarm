function [xdot,A] = swarm_dynamics(t,x,t_ephem,x_ephem)

Nsat = length(x)/6;

xdot = zeros(length(x),1);

if nargout == 2
    A = zeros(length(x),length(x));
end

for k = 1:Nsat
    if nargout == 2
        [xdsat,Asat] = dynamics(t,x,t_ephem,x_ephem);
        xdot(6*(k-1)+(1:6)) = xdsat;
        A(6*(k-1)+(1:6),6*(k-1)+(1:6)) = Asat;
    else
        [xdsat] = dynamics(t,x,t_ephem,x_ephem);
        xdot(6*(k-1)+(1:6)) = xdsat;
    end
end

end

