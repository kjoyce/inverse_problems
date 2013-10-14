function[calpha] = curvatureLcurve(alpha,A,U,S,V,b)

% Compute minus the curvature of the L-curve
dS = diag(S); dS2 = dS.^2;
Utb = U'*b;
xalpha = V*((dS./(dS2+alpha)).*(Utb));
ralpha = A*xalpha-b;
xi = norm(xalpha)^2;
rho = norm(ralpha)^2;

% From Hansen 2010 -- seems to be incorrect.
%zalpha = V*((dS./(dS2+alpha)).*(U'*ralpha));
%xi_p = (4/sqrt(alpha))*xalpha'*zalpha;
%calpha =   2*(xi*rho/xi_p) *...
%           (alpha*xi_p*rho+2*sqrt(alpha)*xi*rho+alpha^2*xi*xi_p) / ...
%           (alpha*xi^2+rho^2)^(3/2);

% From Vogel 2002. 
xi_p = sum(-2*dS2.*Utb.^2./(dS2+alpha).^3);
calpha = - ( (rho*xi)*(alpha*rho+alpha^2*xi)+(rho*xi)^2/xi_p ) / ...
           ( rho^2+alpha^2*xi^2)^(3/2);