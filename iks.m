function [xhist,Phist] = iks(x0,P0,Q,R,dt,thist,yhist,xhub,t_ephem,x_ephem)

%Setup
N = length(thist);
Nx = length(x0);
Ny = size(yhist,1);
Qinv = inv(Q);
Rinv = inv(R);

xhist = zeros(Nx,N);
zhist = zeros(Ny,N);
A = zeros(Nx,Nx,N);
C = zeros(Ny,Nx,N);

%Initial Rollout
xhist(:,1) = x0;
[yp,C(:,:,1)] = observation(x0,xhub(:,1));
zhist(:,1) = yhist(:,1) - yp;
J = zhist(:,1)'*Rinv*zhist(:,1);
for k = 1:(N-1)
    [xhist(:,k+1), A(:,:,k)] = ode3(thist(k),xhist(:,k));
    [yp,C(:,:,k+1)] = observation(xhist(:,k+1),xhub(:,k+1));
    zhist(:,k+1) = yhist(:,k+1) - yp;
    J = J + zhist(:,k+1)'*Rinv*zhist(:,k+1);
end

dx = 1;
alpha = 1;
while max(max(abs(alpha*dx))) > 1e-8
    
    [dxhat,Phat,dxbar,Pbar] = forwardpass(A,C,P0,zhist);
    [dx,Phist] = backpass(A,dxhat,Phat,dxbar,Pbar);
    [xhist,zhist,A,C,J,alpha] = rollout(xhist,dx,inv(Phist(:,:,1)),Qinv,Rinv,J);
    
end

%--------------- Rollout ---------------%
    function [xnew,znew,A,C,Jnew,alpha] = rollout(xhist,dx,P0inv,Qinv,Rinv,J)
        
        %Setup
        xnew = zeros(Nx,N);
        znew = zeros(Ny,N);
        A = zeros(Nx,Nx,N);
        C = zeros(Ny,Nx,N);
        
        %step size
        alpha = 2;
        dJ = 1;
        while dJ > 0 && alpha > 1e-7
            alpha = alpha/2;
            xnew(:,1) = xhist(:,1) + alpha*dx(:,1);
            [yp,C(:,:,1)] = observation(xnew(:,1),xhub(:,1));
            znew(:,1) = yhist(:,1) - yp;
            Jnew = alpha*alpha*dx(:,1)'*P0inv*dx(:,1) + znew(:,1)'*Rinv*znew(:,1);
            
            for k = 1:(N-1)
                
                [xp, A(:,:,k)] = ode3(thist(k),xnew(:,k));
                xnew(:,k+1) = xhist(:,k+1) + alpha*dx(:,k+1);
                
                [yp,C(:,:,k+1)] = observation(xnew(:,k+1),xhub(:,k+1));
                znew(:,k+1) = yhist(:,k+1) - yp;
                
                Jnew = Jnew + (xnew(:,k+1) - xp)'*Qinv*(xnew(:,k+1) - xp) + znew(:,k+1)'*Rinv*znew(:,k+1);
            end
            
            dJ = Jnew - J;
        end
    end

%--------------- KF Forward Pass ---------------%
    function [dxhat,Phat,dxbar,Pbar] = forwardpass(A,C,P0,zhist)
        
        %Setup
        dxbar = zeros(Nx,N);
        dxhat = dxbar;
        Pbar = zeros(Nx,Nx,N);
        Pbar(:,:,1) = P0;
        Phat = Pbar;
        
        %Innovation Covariance
        S = C(:,:,1)*P0*C(:,:,1)' + R;
        
        %Kalman Gain
        K = (P0*C(:,:,1)')/S;
        
        %Update
        dxhat(:,1) = K*zhist(:,1);
        Phat(:,:,1) = (eye(Nx)-K*C(:,:,1))*P0*(eye(Nx)-K*C(:,:,1))' + K*R*K';
        
        for k = 1:(N-1)
            %Predict State
            dxbar(:,k+1) = A(:,:,k)*dxhat(:,k);
            Pbar(:,:,k+1) = A(:,:,k)*Phat(:,:,k)*A(:,:,k)' + Q;
            
            %Innovation
            z = zhist(:,k+1) - C(:,:,k+1)*dxbar(:,k+1);
            S = C(:,:,k+1)*Pbar(:,:,k+1)*C(:,:,k+1)' + R;
            
            %Kalman Gain
            K = (Pbar(:,:,k+1)*C(:,:,k+1)')/S;
            
            %Update
            dxhat(:,k+1) = dxbar(:,k+1) + K*z;
            Phat(:,:,k+1) = (eye(Nx)-K*C(:,:,k+1))*Pbar(:,:,k+1)*(eye(Nx)-K*C(:,:,k+1))' + K*R*K';
        end
    end

%--------------- RTS Backward Pass ---------------%
    function [dx,Phist] = backpass(A,dxhat,Phat,dxbar,Pbar)
        dx = dxhat;
        Phist = Phat;
        for k = (N-1):-1:1
            L = Phat(:,:,k)*A(:,:,k)'/Pbar(:,:,k+1);
            dx(:,k) = dxhat(:,k) + L*(dx(:,k+1) - dxbar(:,k+1));
            Phist(:,:,k) = Phat(:,:,k) + L*(Phist(:,:,k+1) - Pbar(:,:,k+1))*L';
        end
    end

    function [xn, A] = ode3(t,x)
        [xdot1,A1] = dynamics_mex(t,x,t_ephem,x_ephem);
        [xdot2,A2] = dynamics_mex(t+dt/2,x + (dt/2)*xdot1,t_ephem,x_ephem);
        [xdot3,A3] = dynamics_mex(t+dt,x - dt*xdot1 + 2*dt*xdot2,t_ephem,x_ephem);
        xn = x + (dt/6)*(xdot1 + 4*xdot2 + xdot3);
        A = eye(Nx) + (dt/6)*(A1+4*A2+A3) + (dt*dt/6)*(2*A2*A1-A3*A1+2*A3*A2) + (dt*dt*dt/6)*A3*A2*A1;
    end

end

