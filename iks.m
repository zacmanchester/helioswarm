function [xhat,Phat] = iks(x0,P0,Q,R,thist,yhist,xhub,t_ephem,x_ephem)

%Setup
Nx = length(x0);
xhist = zeros(Nx,length(thist));
xbar = xhist;
xbar(:,1) = x0;
xhat = xbar;
Phist = zeros(Nx,Nx,length(thist));
Pbar = Phist;
Pbar(:,:,1) = P0;
Phat = Pbar;
A = zeros(Nx,Nx,length(thist));
dt = thist(2)-thist(1); %assume uniform sampling


%--------------- Forward Pass ---------------%

%First step
xnew = x0;
dx = 1;
% while any(abs(dx) > 1e-6)
    [yp, C] = observation(xnew,xhub(:,1));
    
    %Innovation
    z = yhist(:,1) - yp;
    S = C*P0*C' + R;
    
    %Kalman Gain
    K = (P0*C')/S;
    
    %Update
    dx = K*z;
    xnew = xnew + dx;
% end
xhist(:,1) = xnew;
Phist(:,:,1) = (eye(Nx)-K*C)*P0*(eye(Nx)-K*C)' + K*R*K';

for k = 1:(length(thist)-1)
    
    %Predict
    [xbar(:,k+1), A(:,:,k)] = ode3(thist(k),xhist(:,k));
    Pbar(:,:,k+1) = A(:,:,k)*Phist(:,:,k)*A(:,:,k)' + Q;
    
    
    xnew = xbar(:,k+1);
    dx = 1;
%     while any(abs(dx) > 1e-6)
        [yp, C] = observation(xnew,xhub(:,k+1));
        
        %Innovation
        z = yhist(:,k+1) - yp;
        S = C*Pbar(:,:,k+1)*C' + R;
        
        %Kalman Gain
        K = (Pbar(:,:,k+1)*C')/S;
        
        %Update
        dx = K*z;
        xnew = xnew + dx;
%     end
    xhist(:,k+1) = xnew;
    Phist(:,:,k+1) = (eye(Nx)-K*C)*Pbar(:,:,k+1)*(eye(Nx)-K*C)' + K*R*K';
    
end

%--------------- Backward Pass ---------------%

xhat(:,end) = xhist(:,end);
Phat(:,:,end) = Phist(:,:,end);
for k = (length(thist)-1):-1:1
    L = Phist(:,:,k)*A(:,:,k)'/Pbar(:,:,k+1);
    xhat(:,k) = xhist(:,k) + L*(xhat(:,k+1) - xbar(:,k+1));
    Phat(:,:,k) = Phist(:,:,k) + L*(Phat(:,:,k+1) - Pbar(:,:,k+1))*L';
end


    function [xn, A] = ode3(t,x)
        [xdot1,A1] = dynamics(t,x,t_ephem,x_ephem);
        [xdot2,A2] = dynamics(t+dt/2,x + (dt/2)*xdot1,t_ephem,x_ephem);
        [xdot3,A3] = dynamics(t+dt,x - dt*xdot1 + 2*dt*xdot2,t_ephem,x_ephem);
        xn = x + (dt/6)*(xdot1 + 4*xdot2 + xdot3);
        A = eye(Nx) + (dt/6)*(A1+4*A2+A3) + (dt*dt/6)*(2*A2*A1-A3*A1+2*A3*A2) + (dt*dt*dt/6)*A3*A2*A1;
    end

end

