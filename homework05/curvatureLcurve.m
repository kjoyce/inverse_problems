function[calpha] = curvatureLcurve(alpha,ahat,b)

% Compute minus the curvature of the L-curve
bhat = fft2(b);
xalpha = real(ifft2((ahat./(abs(ahat).^2+alpha)).*bhat));
ralpha = real(ifft2(ahat.*fft2(xalpha)))-b;
xi = norm(xalpha)^2;
rho = norm(ralpha)^2;

% From Vogel 2002. 
xi_p = sum(sum(-2*abs(ahat).^2.*abs(bhat).^2./(abs(ahat).^2+alpha).^3));
calpha = - ( (rho*xi)*(alpha*rho+alpha^2*xi)+(rho*xi)^2/xi_p ) / ...
           ( rho^2+alpha^2*xi^2)^(3/2);