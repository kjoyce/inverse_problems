% Simulation from a 1D and 2D GMRF
load satellite
%x_true = x_true/max(x_true(:));
[n,n] = size(x_true);
N = n^2;
h = 1/n;
x = [-0.5+h/2:h:0.5-h/2]';
[X,Y]=meshgrid(x);
sig = 0.02;
kernel = exp(-((X-h/2).^2+(Y-h/2).^2)/2/sig^2);
kernel = kernel/sum(sum(kernel));
ahat = fft2(fftshift(kernel));

Ax = feval('Amult',x_true,ahat);
err_lev = 2; %%%input(' Percent error in data = ')
sigma = err_lev/100 * norm(Ax(:)) / sqrt(N);
rng(0)
eta =  sigma * randn(n,n);
b = Ax + eta;
bhat = fft2(b);
figure(1) 
  imagesc(x_true), colorbar, colormap(1-gray)
figure(2)
  imagesc(b), colorbar, colormap(1-gray)

% Construct discrete Laplacian L.
%%
% DFT representation
lh=zeros(n,n); lh(1,1)=-1; lh(1,2)=1; lhhat=fft2(lh);
lv=zeros(n,n); lv(1,1)=-1; lv(n,1)=1; lvhat=fft2(lv);
lhat = abs(lhhat).^2+abs(lvhat).^2;

% Sparse matrix representation
D = spdiags([-ones(n,1) ones(n,1)],[0 1],n,n); D(n,1)=1;
I = speye(n,n); Dh = kron(I,D); Dv = kron(D,I);

% Deblur image using conjugate gradient iteration
% Store preliminary CG iteration information
params.max_cg_iter     = 100;
params.cg_step_tol     = 1e-4;
params.grad_tol        = 1e-4;
params.cg_io_flag      = 0;
params.cg_figure_no    = [];
params.precond         = 'Amult';
Bmult_fn               = 'Bmult_IGMRF';
a_params.ahat          = ahat;
a_params.lhat          = lhat;
a_params.Dh            = Dh;
a_params.Dv            = Dv;
c                      = feval('Amult',b,conj(ahat));

% Compute GCV choice of alpha.
reg_flag = input(' Enter 0 for negative-Laplacian and 1 or independent increment regularization. ');
if reg_flag == 0 
    a_params.Lambda = ones(n^2,1);
elseif reg_flag == 1 
    a_params.Lambda = 1./sqrt((Dh*xalpha(:)).^2+(Dv*xalpha(:)).^2+0.001);
end
params.a_params        = a_params;
RegParam_fn            = @(alpha) GCV_fn2d(alpha,b,c,params,Bmult_fn);
alpha                  = fminbnd(RegParam_fn,0,1);

% Plot the corresponding regularized solution.
a_params.alpha         = alpha;
params.a_params        = a_params;
params.cg_step_tol     = 1e-8;
params.grad_tol        = 1e-8;
params.precond         = 'Amult';
params.precond_params  = 1./(abs(ahat).^2+alpha*lhat);
[xalpha,iter_hist]     = CG(zeros(n,n),c,params,Bmult_fn);

% Output the solution.
figure(3)
  imagesc(xalpha), colorbar, colormap(1-gray)
figure(4)
  semilogy(iter_hist(:,2),'k*'), title('CG iteration vs. residual norm')

nsamps = 500;
fourier_filt  = (1/sigma^2)*abs(ahat).^2 + (alpha/sigma^2)*lhat;
xsamps = zeros(nx*ny,nsamps);
for i=1:nsamps
    h = waitbar(i/nsamps);
    v = randn(n,n);
    rhs = c/sigma^2 + real(ifft2(conj(ahat).*fft2(v/sigma)))...
                    + sqrt(alpha/sigma^2)*real(ifft2(sqrt(lhat).*fft2(v)));
    xtemp = CG(zeros(n,n),c,params,Bmult_fn);
    xsamps(:,i)=xtemp(:);
end
close(h)
figure(4)
  imagesc(reshape(mean(xsamps,2),nx,ny)), colorbar,colormap(1-gray)
  title('Mean image from samples')
figure(5)
  imagesc(reshape(std(xsamps,0,2),nx,ny)), colorbar,colormap(1-gray)
  title('Pixel-wise std of image samples')
