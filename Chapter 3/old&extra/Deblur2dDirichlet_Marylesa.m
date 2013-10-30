%  
%  2d image deblurring inverse problem with Dirichlet boundary conditions.
%
  % Build Fourier representer for the matrix Ahat
  clear all, close all
  load satellite.mat
  [nx,ny]=size(x_true);
  x_true = x_true/max(x_true(:));
  n = nx*ny;
  h = 1/nx;
  x = [-1+h:h:1-h]';
  [X,Y]=meshgrid(x);
  sig = 0.02;
  kernel = exp(-(X.^2+Y.^2)/2/sig^2);
  a_s  = zeros(2*nx,2*ny);
  a_s(2:2*nx,2:2*ny) = kernel;
  a_s_hat = fft2(fftshift(a_s));
  a_s_hat = a_s_hat/a_s_hat(1,1);
  
  % Generate Data
  Ax = Amult_Dirichlet(x_true,a_s_hat);
  err_lev = 2; %%%input(' Percent error in data = ')
  sigma = err_lev/100 * norm(Ax(:)) / sqrt(n);
  rng(0)
  eta =  sigma * randn(nx,ny);
  b = Ax + eta;
  figure(1)
    imagesc(x_true), colorbar, colormap(1-gray)
  figure(2)
    imagesc(b), colorbar, colormap(1-gray)
    
  % Deblur image using conjugate gradient iteration
  % Store preliminary CG iteration information
  params.max_cg_iter     = 100;   
  params.cg_step_tol     = 1e-5;    
  params.grad_tol        = 1e-5;  
  params.cg_io_flag      = 0;  
  params.cg_figure_no    = [];
  params.precond         = 'Amult_Dirichlet';
  Bmult_fn               = 'Bmult_Dirichlet';
  a_params.a_s_hat       = a_s_hat;
  params.a_params        = a_params;
  c                      = Amult_Dirichlet(b,conj(a_s_hat));
  
  % Decide on a regularization parameter selection method.
  alpha_flag = input(' Enter 1 for GCV, 2 for DP, or 3 for L-Curve regularization parameter selection. ');
  if alpha_flag == 1
    RegParam_fn            = @(alpha) GCV_fn(alpha,b,c,params,Bmult_fn);
  elseif alpha_flag == 2
    RegParam_fn            = @(alpha) (DP_fn(alpha,b,c,params,Bmult_fn,sigma))^2;
  elseif alpha_flag == 3
    RegParam_fn            = @(alpha) Lcurve_fn(alpha,b,c,params,Bmult_fn);
  end
  alpha                  = fminbnd(RegParam_fn,0,1);
  
  % Compute regularized solution for the chosen alpha.
  a_params.alpha         = alpha;
  params.a_params        = a_params;
  params.precond_params  = 1./(abs(a_s_hat).^2+alpha);
  xalpha                 = CG(zeros(nx,ny),c,params,Bmult_fn);

  % Output the solution.
  figure(3)
    imagesc(xalpha), colorbar, colormap(1-gray)