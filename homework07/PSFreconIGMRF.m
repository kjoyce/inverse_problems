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

% second derivative precision matrix for prior
  D = spdiags([-ones(n,1) ones(n,1)],[0 1],n-1,n);
%  niters = input(' Enter # of iterations of GMRF edge-preserving reconstruction algorithm. ');
  niters = 100;
  last_dp = 1e5;
  for i=1:niters
      if i==1, G = speye(n-1);
      else G = diag(1./sqrt((D*xalpha).^2+.001));
      end
      L = D'*G*D;
      % Compute GCV choice of alpha
      alpha = fminbnd(@(alpha) GCV_fn(alpha,A,L,b),0,1);
      xalpha = (A'*A+alpha*L^4)\(A'*b);
      dp_ratio = norm(A*xalpha-b)^2/(n*sigma^2);
      figure(2)
        plot(t,x_true,'b',t,xalpha,'k','LineWidth',1)
        legend('true image','blurred, noisy data')
        %pause(0.1), 
	title(sprintf('iteration = %d\n',i))
	drawnow
      fprintf('iteration = %d, dp_ratio =%.3f\n',i,dp_ratio)
      %if dp_ratio < 1, break; end;
      if abs(last_dp-dp_ratio)/dp_ratio < 1e-3, break; end;
      last_dp = dp_ratio;
end
