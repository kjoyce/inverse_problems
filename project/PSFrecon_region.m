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
  D = spdiags([-ones(n,1) ones(n,1)],[0 1],n-1,n);

%%% These were obtained from PSFreconIGMRF
% Run this code after PSFreconIGMRF.m
% idx = find(abs(D*xalpha) > 1)
% idx(0) - 2
% idx(1) + 10
% Maybe the way forward is to iterate so that the tails have TV?

n1 = 34;
n2 = 55;
%%%%%%%%%%%%%%%%

alpha = fminbnd(@(alpha) GCV_fn(alpha,A,L,b),0,1);

% Just guessing for now
del1 = 1e5*alpha;
del2 = alpha;
del3 = 1e5*alpha;

% Construct W
dW = ones(n,1)*del2;
dW(1:n1) = del1;
dW(n2:end) = del3;
W = diag(dW);
xalpha = (A'*A+W*L)\(A'*b);

figure(2) 
  plot(t,x_true,'b',t,xalpha,'k','LineWidth',1)
  legend('true image','blurred, noisy data'), ylim([-1,13])

% What does convolution of delta mean in Bayesian terms?
% Convolve W with a square wave twice so it's a little smoother
kern2 = abs(t) < 5e-2; 
%WW = diag(h*toeplitz(+kern2)^2 * diag(W));
WW = h*toeplitz(+kern2)^2 * W;

xxalpha = (A'*A+WW*L)\(A'*b);
figure(3) 
  plot(t,x_true,'b',t,xxalpha,'k','LineWidth',1)
  legend('true image','blurred, noisy data'), ylim([-1,13])
