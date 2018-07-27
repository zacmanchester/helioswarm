function [xhist,Phist] = iukf(x0,P0,Q,R,thist,yhist,t_ephem,x_ephem)

%Setup
Nx = length(x0);
Nv = length(Q);
Nw = length(R);

%Tuning parameters for sigma points
alpha = .05;
beta = 2; %2 is optimal for a Gaussian distribution
kappa = 3-Nx; %choose kappa so that kappa + Nx = 3

lambda_p = alpha^2*(kappa + Nx + Nv) - Nx - Nv;
lambda_u = alpha^2*(kappa + Nx + Nw) - Nx - Nw;

%Compute prediction sigma point weights
w_pm = 1/(2*(Nx+Nv+lambda_p))*ones(2*(Nx+Nv)+1,1);
w_pm(1) = lambda_p/(Nx+Nv+lambda_p);
w_pc = w_pm;
w_pc(1) = lambda_p/(Nx+Nv+lambda_p) + 1 - alpha^2 + beta^2;

%Compute update sigma point weights
w_um = 1/(2*(Nx+Nw+lambda_u))*ones(2*(Nx+Nw)+1,1);
w_um(1) = lambda_u/(Nx+Nw+lambda_u);
w_uc = w_um;
w_uc(1) = lambda_u/(Nx+Nw+lambda_u) + 1 - alpha^2 + beta^2;

%Initialize Arrays
xhist = zeros(Nx,length(thist));
Phist = zeros(Nx,Nx,length(thist));
dt = thist(2)-thist(1); %assume uniform sampling

%-------------------- First Step --------------------%
%Generate prediction sigma points
S = chol(blkdiag(P0,Q))';
chi_p = zeros(Nx+Nv,2*(Nx+Nv)+1);
chi_p(:,1) = [x0;zeros(Nv,1)];
for j = 1:(Nx+Nv)
    chi_p(:,1+j) = chi_p(:,1) + sqrt(Nx+Nv+lambda_p)*S(:,j);
    chi_p(:,1+Nx+Nv+j) = chi_p(:,1) - sqrt(Nx+Nv+lambda_p)*S(:,j);
end

%Propagate sigma points
x_p = zeros(Nx,2*(Nx+Nv)+1);
for j = 1:(2*(Nx+Nv)+1)
    x_p(:,j) = chi_p(1:Nx,j) + chi_p((Nx+1):(Nx+Nv),j);
end

%Calculate predicted state
xbar = zeros(Nx,1);
for j = 1:length(w_pm)
    xbar = xbar + w_pm(j)*x_p(:,j);
end
Pbar = zeros(Nx,Nx);
for j = 1:length(w_pc)
    Pbar = Pbar + w_pc(j)*(x_p(:,j) - xbar)*(x_p(:,j) - xbar)';
end

xnew = xbar;
dx = 1;
while any(abs(dx) > 1e-3)
    
    %Generate update sigma points
    chi_u = zeros(Nx+Nw,2*(Nx+Nv)+1);
    chi_u(:,1) = [xnew;zeros(Nw,1)];
    S = chol(blkdiag(P0,R))';
    for j = 1:(Nx+Nw)
        chi_u(:,1+j) = chi_u(:,1) + sqrt(Nx+Nw+lambda_u)*S(:,j);
        chi_u(:,1+Nx+Nw+j) = chi_u(:,1) - sqrt(Nx+Nw+lambda_u)*S(:,j);
    end
    for j = (2*(Nx+Nw)+1):(2*(Nx+Nv)+1)
        chi_u(:,j) = mean(chi_u(:,(1:(Nx+Nw)))')';
    end
    
    %Calculate predicted measurements
    y_p = zeros(Nw,(2*(Nx+Nv)+1));
    for j = 1:(2*(Nx+Nv)+1)
        y_p(:,j) = observation(chi_u(1:Nx,j),chi_u((Nx+1):(Nx+Nw),j));
    end
    ybar = zeros(Nw,1);
    for j = 1:length(w_um)
        ybar = ybar + w_um(j)*y_p(:,j);
    end
    Pyy = zeros(Nw,Nw);
    for j = 1:length(w_uc)
        Pyy = Pyy + w_uc(j)*(y_p(:,j) - ybar)*(y_p(:,j) - ybar)';
    end
    Pxy = zeros(Nx,Nw);
    for j = 1:length(w_pc)
        Pxy = Pxy + w_pc(j)*(x_p(:,j) - xbar)*(y_p(:,j) - ybar)';
    end
    
    %Innovation
    z = yhist(:,1) - ybar;
    
    %Kalman Gain
    K = Pxy/Pyy;
    
    %Update
    dx = K*z;
    xnew = xnew + dx;
end
xhist(:,1) = xnew;
Phist(:,:,1) = Pbar - K*Pxy';

for k = 1:(length(thist)-1)
    
%Generate prediction sigma points
S = chol(blkdiag(Phist(:,:,k),Q))';
chi_p = zeros(Nx+Nv,2*(Nx+Nv)+1);
chi_p(:,1) = [xhist(:,k); zeros(Nv,1)];
for j = 1:(Nx+Nv)
    chi_p(:,1+j) = chi_p(:,1) + sqrt(Nx+Nv+lambda_p)*S(:,j);
    chi_p(:,1+Nx+Nv+j) = chi_p(:,1) - sqrt(Nx+Nv+lambda_p)*S(:,j);
end

%Propagate sigma points
x_p = zeros(Nx,2*(Nx+Nv)+1);
for j = 1:(2*(Nx+Nv)+1)
    x_p(:,j) = ode4(thist(k), chi_p(1:Nx,j), chi_p((Nx+1):(Nx+Nv),j));
end

%Calculate predicted state
xbar = zeros(Nx,1);
for j = 1:length(w_pm)
    xbar = xbar + w_pm(j)*x_p(:,j);
end
Pbar = zeros(Nx,Nx);
for j = 1:length(w_pc)
    Pbar = Pbar + w_pc(j)*(x_p(:,j) - xbar)*(x_p(:,j) - xbar)';
end

xnew = xbar;
dx = 1;
while any(abs(dx) > 1e-3)
    
    %Generate update sigma points
    chi_u = zeros(Nx+Nw,2*(Nx+Nv)+1);
    chi_u(:,1) = [xnew;zeros(Nw,1)];
    S = chol(blkdiag(Pbar,R))';
    for j = 1:(Nx+Nw)
        chi_u(:,1+j) = chi_u(:,1) + sqrt(Nx+Nw+lambda_u)*S(:,j);
        chi_u(:,1+Nx+Nw+j) = chi_u(:,1) - sqrt(Nx+Nw+lambda_u)*S(:,j);
    end
    for j = (2*(Nx+Nw)+1):(2*(Nx+Nv)+1)
        chi_u(:,j) = mean(chi_u(:,(1:(Nx+Nw)))')';
    end
    
    %Calculate predicted measurements
    y_p = zeros(Nw,(2*(Nx+Nv)+1));
    for j = 1:(2*(Nx+Nv)+1)
        y_p(:,j) = observation(chi_u(1:Nx,j),chi_u((Nx+1):(Nx+Nw),j));
    end
    ybar = zeros(Nw,1);
    for j = 1:length(w_um)
        ybar = ybar + w_um(j)*y_p(:,j);
    end
    Pyy = zeros(Nw,Nw);
    for j = 1:length(w_uc)
        Pyy = Pyy + w_uc(j)*(y_p(:,j) - ybar)*(y_p(:,j) - ybar)';
    end
    Pxy = zeros(Nx,Nw);
    for j = 1:length(w_pc)
        Pxy = Pxy + w_pc(j)*(x_p(:,j) - xbar)*(y_p(:,j) - ybar)';
    end
    
    %Innovation
    z = yhist(:,k+1) - ybar;
    
    %Kalman Gain
    K = Pxy/Pyy;
    
    %Update
    dx = K*z;
    xnew = xnew + dx;
end
xhist(:,k+1) = xnew;
Phist(:,:,k+1) = Pbar - K*Pxy';
    
end

    function xn = ode4(t,x,v)
        xdot1 = swarm_dynamics(t,x,t_ephem,x_ephem);
        xdot2 = swarm_dynamics(t+.5*dt,x+.5*dt*xdot1,t_ephem,x_ephem);
        xdot3 = swarm_dynamics(t+.5*dt,x+.5*dt*xdot2,t_ephem,x_ephem);
        xdot4 = swarm_dynamics(t+dt,x+dt*xdot3,t_ephem,x_ephem);
        xn = x + (dt/6)*(xdot1 + 2*xdot2 + 2*xdot3 + xdot4) + v;
    end

end
