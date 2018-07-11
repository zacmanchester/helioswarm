function [y,C] = observation(x,w)

N = length(x)/6;

y = zeros(3 + (N-1),1);

y(1:3) = x(1:3);

for k = 1:(N-1)
    y(3+k) = norm(x(k*6+(1:3))-x(1:3));
end

if nargin == 2
    y = y + w;
end

if nargout == 2
    C = zeros(length(y),length(x));
    C(1:3,1:3) = eye(3);
    for k = 1:(N-1)
        C(3+k,:) = [(x(1:3)-x(k*6+(1:3)))'/y(3+k), zeros(1,3), zeros(1,6*(k-1)), (x(k*6+(1:3))-x(1:3))'/y(3+k), zeros(1,3), zeros(1,6*(N-1-k))];
    end
    
end

end

