%  
%  1d image deblurring inverse problem with Dirichlet boundary conditions.
%
clear all, close all
n = 80; %%%input(' No. of grid points = ');
h = 1/n;
t = [h/2:h:1-h/2]';
sig = .05; %%%input(' Kernel width sigma = ');
kernel = (1/sqrt(pi)/sig) * exp(-(t-h/2).^2/sig^2);
A = toeplitz(kernel)*h;

% Set up true solution x_true and data b = A*x_true + error.
x_true = .75*(.1<t&t<.25) + .25*(.3<t&t<.32) + (.5<t&t<1).*sin(2*pi*t).^4;
x_true = x_true/norm(x_true);
Ax = A*x_true;
err_lev = 2; %%%input(' Percent error in data = ');
sigma = err_lev/100 * norm(Ax) / sqrt(n);
rng(0) % Use randn('seed',0) for old version of MATLAB
eta =  sigma * randn(n,1);
b = Ax + eta;
figure(1), 
  plot(t,x_true,'k',t,b,'ko')
  %legend('true image','blurred, noisy data','Location','NorthWest')

[U,S,V] = svd(A);
dS = diag(S); dS2 = dS.^2; 
Utb = U'*b;

% Visualize the L-curve and its curvature.
alpha_vec = logspace(-6,-1);
xi_vals = zeros(length(alpha_vec),1);
rho_vals = zeros(length(alpha_vec),1);
curv_vals = zeros(length(alpha_vec),1);
for i = 1:length(alpha_vec)
    xalpha = V*((dS./(dS2+alpha_vec(i))).*Utb);
    xi_vals(i) =  norm(xalpha)^2;
    rho_vals(i) = norm(A*xalpha-b)^2;
    c_vals(i) = curvatureLcurve(alpha_vec(i),A,U,S,V,b); 
end
figure(2), 
  loglog(rho_vals,xi_vals,'k'), hold on, 
  loglog(norm(A*xalpha-b)^2,norm(xalpha)^2,'ko'), hold off
  title('L-curve')
figure(3), semilogx(alpha_vec,c_vals), title('Curvature')

% Compute the L-curve value for alpha and the regularized solution.
alpha = fminbnd(@(alpha) - curvatureLcurve(alpha,A,U,S,V,b),0,1);
xalpha = V*((dS./(dS2+alpha)).*(U'*b));
figure(4), plot(t,x_true,'b',t,xalpha,'k')