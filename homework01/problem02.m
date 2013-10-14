n = 100;
a = 0;
b = pi/2;
h = (b-a)/n;

s = (1/(2*n)):1/n:(1-1/(2*n));	    % Quadrature midpoints on unit interval
s = s*(b-a) + a;		    % Scale to given interval
approx = sum( cos(s)*h ); % Integrate

err = abs((sin(b)-sin(a)) - approx)	  % Calculate error
err_bound = sum( cos(s-h/2) )/24e6*(pi/2)^3     % Calculate upper bound for error
