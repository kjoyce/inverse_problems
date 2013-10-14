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

% Compute Tikhonov solution
[U,S,V] = svd(A);
dS = diag(S); dS2 = dS.^2; 
Utb = U'*b;
param_choice = input(' Enter 0 to enter k, 1 for UPRE, 2 for GCV, 3 for DP, or 4 for L-curve. ');
if param_choice == 0
    alpha = input(' alpha = ');
elseif param_choice == 1
    RegParam_fn = @(a) sum((a^2*Utb.^2)./(dS2+a).^2)+2*sigma^2*sum(dS2./(dS2+a));
elseif param_choice == 2
    RegParam_fn = @(a) sum((a^2*Utb.^2)./(dS2+a).^2)/(n-sum(dS2./(dS2+a)))^2;
elseif param_choice == 3
    RegParam_fn = @(a) (sum((a^2*Utb.^2)./(dS2+a).^2)-n*sigma^2)^2;
elseif param_choice == 4
    RegParam_fn = @(alpha) - curvatureLcurve(alpha,A,U,S,V,b);
end
alpha = fminbnd(RegParam_fn,0,1);

% Now compute the regularized solution for TSVD
dSfilt = dS./(dS.^2+alpha);
xfilt = V*(dSfilt.*(U'*b));
figure(2)
  plot(t,x_true,'b-',t,xfilt,'k-')
