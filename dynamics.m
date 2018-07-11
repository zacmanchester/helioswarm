function [xdot, A] = dynamics(t,x,t_ephem,x_ephem)

%Gravitational parameters (GM) in 1000*km^3/day^2
mu_s = 1.3271244004193938e11*((60*60*24)^2/(1000^3));
mu_e = 398600.440*((60*60*24)^2/(1000^3));
mu_m = 4902.80007*((60*60*24)^2/(1000^3));
%J2 = 1.08263e-3; %J2 spherical harmonic coefficient
%Re = 6378.1363/1000; %1000*km

rsm = ephemInterp(t,t_ephem,x_ephem);
r_sun = rsm(1:3);
r_moon = rsm(4:6);

r = x(1:3);
rmag = sqrt(r'*r);
rm = r-r_moon;
rmmag = sqrt(rm'*rm);
rs = r-r_sun;
rsmag = sqrt(rs'*rs);
rfmag = sqrt(r_sun'*r_sun);

v = x(4:6);

a_earth = -mu_e*r/(rmag^3);
a_moon = -mu_m*rm/(rmmag^3);
a_sun = -mu_s*rs/(rsmag^3);
a_frame = -mu_s*r_sun/(rfmag^3);

a = a_earth + a_moon + a_sun + a_frame;
%a = a_earth + a_sun + a_frame;
%a = a_earth + a_moon;
%a = a_earth;

xdot = [v; a];

if nargout == 2
    da_earth = -(mu_e/(rmag^3))*eye(3) + (3*mu_e/(rmag^5))*(r*r');
    da_moon = -(mu_m/(rmmag^3))*eye(3) + (3*mu_m/(rmmag^5))*(rm*rm');
    da_sun = -(mu_s/(rsmag^3))*eye(3) + (3*mu_s/(rsmag^5))*(rs*rs');
    da = da_earth + da_moon + da_sun;
    
    A = [zeros(3), eye(3);
           da,  zeros(3)];
end
     

end

