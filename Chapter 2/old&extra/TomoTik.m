%  
%  1d image deblurring inverse problem with Dirichlet boundary conditions.
%
clear all, close all
n = 80; %%%input(' No. of grid points = ');
h = 1/n;
t = [0:h:1-h]';
A = h*triu(ones(n,n));
  
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
  %legend('true image','blurred, noisy data','Location','NorthWest')
  
% Compute TSVD solution
[U,S,V] = svd(A);
dS = diag(S); 
alpha = 0.0005; % regularization parameter
phi = dS.^2./(dS.^2+alpha); 
idx = (dS>0);
dSfilt = zeros(size(dS));
dSfilt(idx) = phi(idx)./dS(idx); 
xfilt = V*(dSfilt.*(U'*b));
figure(2)
  plot(t,x_true,'b-',t,xfilt,'k-')
