function [y,C] = observation(x,xhub)

y = norm(x(1:3)-xhub(1:3));

if nargout == 2
    C = [(x(1:3)-xhub(1:3))'/y, zeros(1,3)];
end

end

