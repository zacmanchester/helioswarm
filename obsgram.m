function G = obsgram(thist,xhist,dynamics,observation)

Nx = size(xhist,1);
Nsat = Nx/6;
Ny = Nsat+2;
Nt = size(xhist,2);

dt = thist(2)-thist(1); %assume constant time steps

Phi = eye(Nx);
Alast = [];
for m = 1:Nsat
    [~,A] = dynamics(thist(1),xhist(6*(m-1)+(1:6),1));
    Alast = blkdiag(Alast,A);
end

G = zeros(Nx,Nx);

for k = 1:Nt
    [~,C] = observation(xhist(:,k));
    G = G + Phi'*(C'*C)*Phi;
    
    Anow = [];
    for m = 1:Nsat
        [~,A] = dynamics(thist(k),xhist(6*(m-1)+(1:6),k));
        Anow = blkdiag(Anow,A);
    end
    
    Phi = expm(Anow*dt/2)*expm(Alast*dt/2)*Phi;
    Alast = Anow;
end


end

