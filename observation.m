function [y,C] = observation(xnode,xhub)

y = norm(xnode(1:3)-xhub(1:3));

if nargout == 2
    C = [(xnode(1:3)-xhub(1:3))'/y, zeros(1,3)];
end

end

