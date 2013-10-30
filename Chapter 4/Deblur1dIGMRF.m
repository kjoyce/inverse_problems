%  
%  1d image deblurring inverse problem with Dirichlet boundary conditions.
%
  %clear all, close all
  rng(0) % randn('seed',0) for older version of MATLAB & for IPSE figures
  n = 80; %%%input(' No. of grid points = ');
  h = 1/n;
  t = [h/2:h:1-h/2]';
  sig = .05; %%%input(' Kernel width sigma = ');
  kernel = (1/sqrt(pi)/sig) * exp(-(t-h/2).^2/sig^2);
  A = toeplitz(kernel)*h;
  
% Set up true solution x_true and data b = A*x_true + error.
  x_true = 50*(.75*(.1<t&t<.25) + .25*(.3<t&t<.32) + (.5<t&t<1).*sin(2*pi*t).^4);
  Ax = A*x_true;
  err_lev = 2; %%%input(' Percent error in data = ');
  sigma = err_lev/100 * norm(Ax) / sqrt(n);
  eta =  sigma * randn(n,1);
  b = Ax + eta;
  figure(1), 
    plot(t,x_true,'k',t,b,'ko','LineWidth',1)
    legend('true image','blurred, noisy data')
  
% second derivative precision matrix for prior
  D = spdiags([-ones(n,1) ones(n,1)],[0 1],n-1,n);
  niters = input(' Enter # of iterations of GMRF edge-preserving reconstruction algorithm. ');
  for i=1:niters
      if i==1, G = speye(n-1);
      else G = diag(1./sqrt((D*xalpha).^2+.001));
      end
      L = D'*G*D;
      % Compute GCV choice of alpha
      alpha = fminbnd(@(alpha) GCV_fn(alpha,A,L,b),0,1);
      xalpha = (A'*A+alpha*L)\(A'*b);
      figure(2)
        plot(t,x_true,'b',t,xalpha,'k','LineWidth',1)
        legend('true image','blurred, noisy data')
        pause(0.1), drawnow
      fprintf('iteration = %d\n',i)
end