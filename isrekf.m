function [xhist,Uhist] = isrekf(x0,U0,V,W,thist,yhist,xhub,t_ephem,x_ephem)

%Setup
Nx = length(x0);
Ny = size(yhist,1);
xhist = zeros(Nx,length(thist));
Uhist = zeros(Nx,Nx,length(thist));
dt = thist(2)-thist(1); %assume uniform sampling

% V = chol(Q);
% W = chol(R);
% U0 = chol(P0);

%First step
xnew = x0;
dx = 1;
while any(abs(dx) > 1e-6)
    [yp, C] = observation(xnew,xhub(:,1));
    
    %Innovation
    z = yhist(:,1) - yp;
    S = qr([U0*C'; W]);
    S = triu(S(1:Ny,1:Ny));
    
    %Kalman Gain
    K = ((U0'*U0*C')/S)/(S');
    
    %Update
    dx = K*z;
    xnew = xnew + dx;
end
xhist(:,1) = xnew;
Unew = qr([U0*(eye(Nx)-K*C)'; W*K']);
Uhist(:,:,1) = triu(Unew(1:Nx,1:Nx));

for k = 1:(length(thist)-1)
    
    %Predict
    [xp, A] = ode3(thist(k),xhist(:,k));
    Up = qr([Uhist(:,:,k)*A'; V]);
    Up = triu(Up(1:Nx,1:Nx));
    
    xnew = xp;
    dx = 1;
    while any(abs(dx) > 1e-6)
        [yp, C] = observation(xnew,xhub(:,k+1));
        
        %Innovation
        z = yhist(:,k+1) - yp;
        S = qr([Up*C'; W]);
        S = triu(S(1:Ny,1:Ny));
        
        %Kalman Gain
        K = ((Up'*Up*C')/S)/(S');
        
        %Update
        dx = K*z;
        xnew = xnew + dx;
    end
    
    %Covariance update
    xhist(:,k+1) = xnew;
    Unew = qr([Up*(eye(Nx)-K*C)'; W*K']);
    Uhist(:,:,k+1) = triu(Unew(1:Nx,1:Nx));
    
end

    function [xn, A] = ode2(t,x)
        [xdot1,A1] = dynamics(t,x,t_ephem,x_ephem);
        xmid = x + (dt/2)*xdot1;
        [xdot2,A2] = dynamics(t+dt/2,xmid,t_ephem,x_ephem);
        xn = x + dt*xdot2;
        A = eye(Nx) + dt*A2 + 0.5*dt*dt*A2*A1;
    end

    function [xn, A] = ode3(t,x)
        [xdot1,A1] = dynamics(t,x,t_ephem,x_ephem);
        [xdot2,A2] = dynamics(t+dt/2,x + (dt/2)*xdot1,t_ephem,x_ephem);
        [xdot3,A3] = dynamics(t+dt,x - dt*xdot1 + 2*dt*xdot2,t_ephem,x_ephem);
        xn = x + (dt/6)*(xdot1 + 4*xdot2 + xdot3);
        A = eye(Nx) + (dt/6)*(A1+4*A2+A3) + (dt*dt/6)*(2*A2*A1-A3*A1+2*A3*A2) + (dt*dt*dt/6)*A3*A2*A1;
    end

end

