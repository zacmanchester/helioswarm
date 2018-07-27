function [xhist,Uhist] = isrukf(x0,U0,V,W,thist,yhist,xhub,t_ephem,x_ephem)

%Setup
Nx = length(x0);
Ny = length(yhist(:,1));

%Tuning parameters for sigma points
alpha = .05;
beta = 2; %2 is optimal for a Gaussian distribution
lambda = Nx*(alpha^2 - 1);
eta = sqrt(Nx+lambda);

%Sigma point weights
w_m0 = lambda/(Nx+lambda);
w_m1 = 1/(2*(Nx+lambda));
w_c0 = lambda/(Nx+lambda) + 1 - alpha^2 + beta;
w_c1 = 1/(2*(Nx+lambda));

%Initialize Arrays
xhist = zeros(Nx,length(thist));
Uhist = zeros(Nx,Nx,length(thist));
dt = thist(2)-thist(1); %assume uniform sampling

%-------------------- First Step --------------------%

%Generate sigma points
chi_x = zeros(Nx,2*Nx+1);
chi_x(:,1) = x0;
for j = 1:Nx
    chi_x(:,1+j) = chi_x(:,1) + eta*U0(:,j);
    chi_x(:,1+Nx+j) = chi_x(:,1) - eta*U0(:,j);
end

%Calculate predicted measurements
chi_y = zeros(Ny,2*Nx+1);
for j = 1:(2*Nx+1)
    chi_y(:,j) = observation(chi_x(:,j),xhub(:,1));
end

ybar = w_m0*chi_y(:,1);
for j = 2:(2*Nx+1)
    ybar = ybar + w_m1*chi_y(:,j);
end

Uyy = qr([sqrt(w_c1)*(chi_y(:,2:end)-ybar)'; W]);
Uyy = triu(Uyy(1:Ny,1:Ny));
Uyy = cholupdate(Uyy,sqrt(-w_c0)*(chi_y(:,1)-ybar),'-');

Pxy = zeros(Nx,Ny);
for j = 2:(2*Nx+1)
    Pxy = Pxy + w_c1*(chi_x(:,j) - x0)*(chi_y(:,j) - ybar)';
end

%Kalman Gain
K = (Pxy/Uyy)/(Uyy');

dx = 1;
xnew = x0;
while any(abs(dx) > 1e-3)
    %Innovation
    z = yhist(:,1) - ybar;
    
    %Update
    dx = K*z;
    xnew = xnew + dx;
    
    ybar = observation(xnew,xhub(:,1));
end
xhist(:,1) = xnew;

UK = K*Uyy';
Unew = U0;
for j = 1:size(UK,2)
    Unew = cholupdate(Unew,UK(:,j),'-');
end
Uhist(:,:,1) = Unew;

for k = 1:(length(thist)-1)
    
    %Generate sigma points
    chi_x = zeros(Nx,2*Nx+1);
    chi_x(:,1) = xhist(:,k);
    for j = 1:Nx
        chi_x(:,1+j) = chi_x(:,1) + eta*Uhist(:,j,k);
        chi_x(:,1+Nx+j) = chi_x(:,1) - eta*Uhist(:,j,k);
    end
    
    %Propagate sigma points
    for j = 1:(2*Nx+1)
        chi_x(:,j) = ode4(thist(k), chi_x(:,j));
    end
    
    %Calculate predicted state
    xbar = w_m0*chi_x(:,1);
    for j = 2:(2*Nx+1)
        xbar = xbar + w_m1*chi_x(:,j);
    end
    
    %Calculate predicted sqrt covariance
    Ubar = qr([sqrt(w_c1)*(chi_x(:,2:end)-xbar)'; V]);
    Ubar = triu(Ubar(1:Nx,1:Nx));
    Ubar = cholupdate(Ubar,sqrt(-w_c0)*(chi_x(:,1)-xbar),'-');
    
    %Calculate predicted measurements
    chi_y = zeros(Ny,2*Nx+1);
    for j = 1:(2*Nx+1)
        chi_y(:,j) = observation(chi_x(:,j),xhub(:,k+1));
    end
    
    ybar = w_m0*chi_y(:,1);
    for j = 2:(2*Nx+1)
        ybar = ybar + w_m1*chi_y(:,j);
    end
    
    Uyy = qr([sqrt(w_c1)*(chi_y(:,2:end)-ybar)'; W]);
    Uyy = triu(Uyy(1:Ny,1:Ny));
    Uyy = cholupdate(Uyy,sqrt(-w_c0)*(chi_y(:,1)-ybar),'-');
    
    Pxy = w_c0*(chi_x(:,1)-xbar)*(chi_y(:,1)-ybar)';
    for j = 2:(2*Nx+1)
        Pxy = Pxy + w_c1*(chi_x(:,j) - xbar)*(chi_y(:,j) - ybar)';
    end
    
    %Kalman Gain
    K = (Pxy/Uyy)/(Uyy');
    
    dx = 1;
    xnew = xbar;
    for j = 1:2
        %Innovation
        z = yhist(:,k+1) - ybar;
        
        %Update
        dx = K*z;
        xnew = xnew + dx;
        
        ybar = observation(xnew,xhub(:,k+1));
    end
    
    xhist(:,k+1) = xnew;
    UK = K*Uyy';
    Unew = Ubar;
    for j = 1:size(UK,2)
        Unew = cholupdate(Unew,UK(:,j),'-');
    end
    Uhist(:,:,k+1) = Unew;
    
end

    function xn = ode4(t,x)
        xdot1 = dynamics_mex(t,x,t_ephem,x_ephem);
        xdot2 = dynamics_mex(t+.5*dt,x+.5*dt*xdot1,t_ephem,x_ephem);
        xdot3 = dynamics_mex(t+.5*dt,x+.5*dt*xdot2,t_ephem,x_ephem);
        xdot4 = dynamics_mex(t+dt,x+dt*xdot3,t_ephem,x_ephem);
        xn = x + (dt/6)*(xdot1 + 2*xdot2 + 2*xdot3 + xdot4);
    end

end
