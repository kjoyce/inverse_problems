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
%param_choice = input(' Enter 0 to enter k, 1 for UPRE, 2 for GCV, 3 for DP, or 4 for L-curve. ');
%if param_choice == 0
%    alpha = input(' alpha = ');
%elseif param_choice == 1
%    RegParam_fn = @(a) sum((a^2*Utb.^2)./(dS2+a).^2)+2*sigma^2*sum(dS2./(dS2+a));
%elseif param_choice == 2
%    RegParam_fn = @(a) sum((a^2*Utb.^2)./(dS2+a).^2)/(n-sum(dS2./(dS2+a)))^2;
%elseif param_choice == 3
%    RegParam_fn = @(a) (sum((a^2*Utb.^2)./(dS2+a).^2)-n*sigma^2)^2;
%elseif param_choice == 4
%    RegParam_fn = @(alpha) - curvatureLcurve(alpha,A,U,S,V,b);
%end
%alpha = fminbnd(RegParam_fn,0,1);


% Now compute the regularized solution for TSVD
%dSfilt = dS./(dS.^2+alpha);
%xfilt = V*(dSfilt.*(U'*b));
%figure(2)
%  plot(t,x_true,'b-',t,xfilt,'k-')

alpha = logspace(-5,0);

% Problem 2.2a
f_a = @(a) norm((A'*A + a*eye(n))\(A'*b) - x_true)/norm(x_true);
rel_error = arrayfun(f_a,alpha);
[rel_min,rel_min_idx] = min(rel_error);
figure(1)
semilogx(alpha,rel_error), hold on;
semilogx(alpha(rel_min_idx),rel_min,'*'), hold off;
xlabel('alpha')
ylabel('relative error')
title(sprintf('Relative error vs. alpha in Tikhonov Solution'));
saveTightFigure(figure(1),'deblur_rel_error.pdf')

figure(2)
plot(t,x_true,'b-', t,(A'*A + alpha(rel_min_idx)*eye(n))\(A'*b),'k-')
title(sprintf('True function with reconstruction \\alpha = %.5f',alpha(rel_min_idx)));
saveTightFigure(figure(2),'psf_rel_error_solution.pdf')

%% TSVD
%x_trunc = @(k) V*blkdiag(eye(n-k),zeros(m,n-k)*S'^(-1))*Utb;
%nu = 1:80;
%x_trunc = @(k) V*diag(1:n <= k)*diag(dS.^(-1))*Utb;
%ff_a = @(k) norm(x_trunc(k) - x_true)/norm(x_true); 
%rrel_error = arrayfun(ff_a,nu);
%[rrel_min,rrel_min_idx] = min(rrel_error);
%figure(2)
%semilogy(nu,rrel_error), hold on;
%semilogy(nu(rrel_min_idx),rrel_min,'*'), hold off;
%xlabel('k');
%ylabel('relative error');
%title(sprintf('TSVD minimum relative error k : %2d',nu(rel_min_idx)));

% Problem 2.5a
figure(3)
m_a = @(a) sigma^2*norm(dS./(dS.^2+a))^2 + (a./(dS.^2 + a))'.^2 * ((V'*x_true).^2); 
mean_error = arrayfun( m_a, alpha);
[mean_min,mean_min_idx] = min(mean_error);
loglog(alpha,mean_error,'r--'), hold on;
loglog(alpha(mean_min_idx),mean_min,'r*'), hold off;
title(sprintf('True function with reconstruction \\alpha = %.5f',alpha(mean_min_idx)));
saveTightFigure(figure(2),'deblur_mse.pdf')

