%  
%  Image deblurring inverse problem with periodic boundary conditions.
%
%% Generate data and build regularization matrix.
  clear all, close all
  load GaussianBlur440_normal
  %load satellite.mat
  x_true = 100*f_true/max(f_true(:)); 
  clear PSF f_true beta g sigma 
  [nx,ny]=size(x_true);
  % Generate data on 256^2 grid w/ periodic BCs, then restrict to 128^2.
  h      = 1/nx;
  x      = [-0.5+h/2:h:0.5-h/2]';
  [X,Y]  = meshgrid(x);
  sig    = 2*h;
  kernel = exp(-((X-h/2).^2+(Y-h/2).^2)/2/sig^2);
  kernel = kernel/sum(sum(kernel));
  ahat   = fft2(fftshift(kernel));
  Ax     = real(ifft2(ahat.*fft2(x_true)));
  % Extract 128x128 subregion from Ax and x, add noise and plot.
  Ax     = Ax(101:228,101:228);
  x_true = x_true(101:228,101:228);
  [nx2,ny2]= size(Ax);
  err_lev= 2;
  noise  = err_lev/100 * norm(Ax(:)) / sqrt(nx2*ny2);
  b      = Ax + noise*randn(nx2,ny2);
  figure(1), imagesc(x_true), colormap(gray), colorbar, 

  % zero pad b and create the mask
  b_pad = padarray(b,[nx/4,ny/4]);
  bp_hat= fft2(b_pad);
  M     = padarray(ones(size(b)),[nx/4,ny/4]);
  figure(2), imagesc(Ax), colormap(gray), colorbar 

  %% Store CG iteration information
  params.max_cg_iter  = 250;   
  params.cg_step_tol  = 1e-4;    
  params.grad_tol     = 1e-4;  
  params.cg_io_flag   = 0;  
  params.cg_figure_no = [];
  params.precond      = 'Amult_circulant';
  params.ahat         = ahat;
  params.M            = M;
  % Store necessary info for matrix vector multiply (A*x) function
  Bmult               = 'Bmult_DataDriven';
  a_params.ahat       = ahat;
  a_params.M          = M;
  params.a_params     = a_params;
  
  % Choose the regularization parameter
  AtDb                = Amult_circulant(b_pad,conj(ahat));
  disp(' *** Computing regularization parameter using GCV *** ')
  RegParam_fn         = @(alpha) GCV_fn(alpha,b,AtDb,params,Bmult);
  alpha               = fminbnd(RegParam_fn,0,1);
  
  %% Use PCG to solve Bx=c.
  % Compute regularized solution for the chosen alpha.
  a_params.alpha         = alpha;
  params.a_params        = a_params;
  params.max_cg_iter     = 500;   
  params.cg_step_tol     = 1e-6;    
  params.grad_tol        = 1e-6;  
  params.precond         = 'Amult_circulant';
  params.precond_params  = 1./(abs(ahat).^2+alpha);
  disp(' *** Computing the regularized solution *** ')
  [xalpha,iter_hist]     = CG(zeros(nx,ny),AtDb,params,Bmult);

  % Plot the results.
  figure(3)
    imagesc(xalpha(nx/4+1:3*nx/4,ny/4+1:3*ny/4),[0,max(x_true(:))]), colorbar, colormap(gray)
  figure(4), 
    semilogy(iter_hist(:,2),'k','LineWidth',2), title('CG iteration vs. residual norm')