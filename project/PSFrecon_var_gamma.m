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
  L = L^2;
% Set hyper-priors for Gamma
gam_alpha = 1;
gam_beta  = .01;
% start with GCV estimate
alpha = fminbnd(@(alpha) GCV_fn(alpha,A,L^(2),b),0,1); 
W = alpha*eye(n);

% Iterative MAP estimate
niter = 100;
for i=1:niter
  xalpha = (A'*A+W*L)\(A'*b);
  figure(3) 
    subplot(1,2,1)
    plot(t,x_true,'b',t,xalpha,'k','LineWidth',1)
    legend('true image','blurred, noisy data'), ylim([-1,13])
    subplot(1,2,2)
    plot(t,diag(W))
    drawnow
  %fprintf('iteration = %d\n norm gam = ',i,norm(diag(W)))
  gamma_hat = gam_alpha/2*( (gam_beta-3/2) + sqrt( (gam_beta - 3/2)^2 + (2*(L*xalpha).^2)/gam_alpha)); 
  %gamma_hat = sqrt(alpha)*gamma_hat;
  %gamma_hat = L*gamma_hat*sqrt(alpha); % make gamma smooth and scale by gcv parameter
  W = diag( gamma_hat );
end
