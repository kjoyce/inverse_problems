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
eta =  sigma * randn(n,1);
b = Ax + eta;
figure(1), 
  plot(t,x_true,'k',t,b,'ko')
  legend('true image','blurred, noisy data','Location','NorthWest')

% Compute TSVD solution
[U,S,V] = svd(A);
dS = diag(S); 

% Create Picard plot.
gen_fourier = abs(U'*b);
scaled_gf = gen_fourier./dS;
figure(2), 
  jj = 1:50;
  semilogy(gen_fourier(jj),'o'), hold on
  semilogy(scaled_gf(jj),'*')
  semilogy(dS(jj),'d'), hold off
  legend('|u_i^T b|','|u_i^T b|/\sigma_i','\sigma_i')

% Compute regularized solution
alpha = 0.01;%input('regularization parameter alpha = ');
dSa = (diagS>alpha)./diagS;
Sa_inv = diag(dSa);
xalpha = V*(Sa_inv*(U'*b));
figure(3)
  plot(t,x_true,'b',t,xalpha,'k')
