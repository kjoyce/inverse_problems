clear all, close all
n       = 80; %%%input(' No. of grid points = ');
h       = 1/n;
t       = [h/2:h:1-h/2]';
% Create the left-half of the PSF
sig1    = .02; %%%input(' Kernel width sigma = ');
kernel1 = exp(-(t(1:40)-h/2).^2/sig1^2);
% Create the right-half of the PSF
sig2    = .08;
kernel2 = exp(-(t(1:40)-h/2).^2/sig2^2);
% Create the normalized PSF
PSF     = [flipud(kernel1);kernel2];
PSF     = PSF/sum(PSF)/h;
% Numerical discretization of intergral operator
A = tril(ones(n,n))*h;

% Set up true solution x_true and data b = A*x_true + error.
x_true = PSF;
Ax = A*x_true;
err_lev = 2; %%%input(' Percent error in data = ');
sigma = err_lev/100 * norm(Ax) / sqrt(n);
%randn('state',1)
eta =  sigma * randn(n,1);
b = Ax + eta;
figure(1), 
  plot(t,x_true,'k',t,b,'ko')

% second derivative precision matrix for prior
  L = spdiags([-ones(n,1) 2*ones(n,1) -ones(n,1)],[-1 0 1],n,n);

% Set hyper-priors for Gamma
  alpha = fminbnd(@(alpha) GCV_fn(alpha,A,L^(1),b),0,1); 
  xalpha = (A'*A+alpha*L^(1))\(A'*b);
  figure(3) 
    plot(t,x_true,'b',t,xalpha,'k','LineWidth',1)
    legend('true image','blurred, noisy data')

N = 10000;
L = L^2;
z = randn(n,N);
w = A'*z + sqrt(alpha)*chol(L')*z;
xalpha = (A'*A + alpha*L)\(repmat(A'*b,[1 N]) + w);

 
  figure(4) 
    plot(t,x_true,'b',t,mean(xalpha,2),'k','LineWidth',1)

