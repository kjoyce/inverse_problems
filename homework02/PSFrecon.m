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
figure(2),
  plot(t,A\b,'k',t,x_true,'b')

delta_b = norm(b-Ax)
delta_x = norm(x_true-A\b)

% SVD analysis
[U,S,V] = svd(A);
figure(3),
  semilogy(diag(S),'ko')

% Singular vector analysis
figure(4)
title('Right Singular Vectors')
for i=1:8
    subplot(2,4,i)
    plot(t,V(:,i))
    title(sprintf('v_{%d}',i))
end
figure(5)
title('Left Singular Vectors')
for i=1:10:80
    subplot(2,4,(i-1)/10+1),
    plot(t,U(:,i)),
    title(sprintf('u_{%d}',i));
end
figure(6),
semilogy(1:n,diag(S),'ko', 1:n,abs(U'*b),'b*', 1:n,abs(U'*b)./diag(S), 'r.')
title('Noisy Data')

figure(7)
semilogy(1:n,diag(S),'ko', 1:n,abs(U'*Ax),'b*', 1:n,abs(U'*Ax)./diag(S), 'r.','Linewidth',2)
title('Data with No Noise')

figure(8)
semilogy(sigma^2./diag(S).^2,'k.')

