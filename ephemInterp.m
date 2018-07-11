function x = ephemInterp(t,tsamp,xsamp)

x1 = interp1(tsamp,xsamp(1,:),t,'linear');
x2 = interp1(tsamp,xsamp(2,:),t,'linear');
x3 = interp1(tsamp,xsamp(3,:),t,'linear');
x4 = interp1(tsamp,xsamp(4,:),t,'linear');
x5 = interp1(tsamp,xsamp(5,:),t,'linear');
x6 = interp1(tsamp,xsamp(6,:),t,'linear');

x = [x1; x2; x3; x4; x5; x6];

end

