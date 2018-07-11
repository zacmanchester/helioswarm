function [xhist,Phist] = iekf(x0,P0,Q,R,thist,yhist,t_ephem,x_ephem)

%Setup
Nx = length(x0);
xhist = zeros(Nx,length(thist));
Phist = zeros(Nx,Nx,length(thist));
dt = thist(2)-thist(1); %assume uniform sampling

%First step
xnew = x0;
dx = 1;
while any(abs(dx) > 1e-5)
    [yp, C] = observation(xnew);
    
    %Innovation
    z = yhist(:,1) - yp;
    S = C*P0*C' + R;
    
    %Kalman Gain
    K = (P0*C')/S;
    
    %Update
    dx = K*z;
    xnew = xnew + dx;
end
xhist(:,1) = xnew;
Phist(:,:,1) = (eye(Nx)-K*C)*P0*(eye(Nx)-K*C)' + K*R*K';

for k = 1:(length(thist)-1)
    
    %Predict
    [xp, A] = ode2(thist(k),xhist(:,k));
    Pp = A*Phist(:,:,k)*A' + Q;
    
    xnew = xp;
    dx = 1;
    while any(abs(dx) > 1e-5)
        [yp, C] = observation(xnew);
        
        %Innovation
        z = yhist(:,k+1) - yp;
        S = C*Pp*C' + R;
        
        %Kalman Gain
        K = (Pp*C')/S;
        
        %Update
        dx = K*z;
        xnew = xnew + dx;
    end
    
    %Covariance update
    xhist(:,k+1) = xnew;
    Phist(:,:,k+1) = (eye(Nx)-K*C)*Pp*(eye(Nx)-K*C)' + K*R*K';
    
end

    function [xn, A] = ode2(t,x)
        [xdot1,A1] = swarm_dynamics(t,x,t_ephem,x_ephem);
        xmid = x + (dt/2)*xdot1;
        [xdot2,A2] = swarm_dynamics(t+dt/2,xmid,t_ephem,x_ephem);
        xn = x + dt*xdot2;
        A = eye(Nx) + dt*A2 + 0.5*dt*dt*A2*A1;
    end

end

