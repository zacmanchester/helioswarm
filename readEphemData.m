function [t,x] = readEphemData(filename)
fileID = fopen(filename);
Data = textscan(fileID,'%f %*c.%*c. %{yyyy-MMM-dd HH:mm:ss.SSSS}D %f %f %f %f %f %f','Delimiter',',','HeaderLines',23);
fclose(fileID);
JulianDate = Data{1};
t = (JulianDate - JulianDate(1)); %time in days
%date = Data{2};
r1 = Data{3}; %km
r2 = Data{4};
r3 = Data{5};
v1 = Data{6}; %km/s
v2 = Data{7};
v3 = Data{8};
x = [r1,r2,r3,v1,v2,v3]';
end