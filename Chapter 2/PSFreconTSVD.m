%  
% PSF reconstruction problem.
%
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
  %legend('true image','blurred, noisy data','Location','NorthWest')

% Compute TSVD solution
[U,S,V] = svd(A);
dS = diag(S); 
Utb = U'*b;
param_choice = input(' Enter 0 to enter k, 1 for UPRE, 2 for GCV, or 3 for DP. ');
if param_choice == 0
    k = input(' k = ');
elseif param_choice == 1
    % Find the UPRE choice for k (see Section 2.2)
    U_fn = @(k) norm(Utb(k+1:n))^2+2*sigma^2*k;
    Uvals = zeros(n,1);
    for i=1:n, Uvals(i)=U_fn(i); end
    k = find(Uvals == min(Uvals));
elseif param_choice == 2
    % Find the GCV choice for k
    G_fn = @(k) norm(Utb(k+1:n))^2/(n-k)^2;
    Gvals = zeros(n,1);
    for i=1:n, Gvals(i)=G_fn(i); end
    k = find(Gvals == min(Gvals));
elseif param_choice == 3
    D_fn = @(k) (norm(Utb(k+1:n))^2-n*sigma^2)^2;
    Dvals = zeros(n,1);
    for i=1:n, Dvals(i)=D_fn(i); end
    k = find(Dvals == min(Dvals));
end

% Now compute the regularized solution for TSVD
phi = zeros(n,1); phi(1:k)=1; 
idx = (dS>0);
dSfilt = zeros(size(dS));
dSfilt(idx) = phi(idx)./dS(idx); 
xfilt = V*(dSfilt.*(U'*b));
%rel_error = norm(xfilt-x_true)/norm(x_true)
figure(2)
  plot(t,x_true,'b-',t,xfilt,'k-')
