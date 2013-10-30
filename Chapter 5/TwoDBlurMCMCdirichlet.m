%  
%  Image deblurring inverse problem with Dirichlet boundary conditions.
%
%% Build Fourier representer for Toeplitz system
clear all, close all
load satellite.mat
rng(0)
[nx,ny] = size(x_true); 
N = nx*ny;
h = 1/nx;
x = [-1+h/2:h:1-h/2]';
[X,Y]=meshgrid(x);
sig = 2*h;
kernel = exp(-((X-h/2).^2+(Y-h/2).^2)/2/sig^2);
kernel = kernel/sum(sum(kernel));
t      = extract(kernel,2*nx-1,2*ny-1);
t_ext  = zeros(2*nx,2*ny);
t_ext(2:2*nx,2:2*ny) = t;
t_ext_hat = fft2(fftshift(t_ext));
  
% Generate Data
Ax=Amult_Dirichlet(x_true,t_ext_hat);
err_lev = 2;
noise = err_lev/100 * norm(Ax(:)) / sqrt(nx*ny);%input(' The standard deviation of the noise = ');
b = Ax + noise*randn(nx,ny);
figure(1), imagesc(b), colormap(1-gray), colorbar
  
%% Construct discrete negative Laplacian L. Assume nx=ny.
Lsub = -ones(nx,1); Ldiag = 2*ones(nx,1);
%Ldiag(1) = 1; Ldiag(nx) = 1; % Neumman B.C.'s
L1 = spdiags([Lsub Ldiag Lsub], [-1 0 1], nx,nx);
%L1(nx,1) = -1; L1(1,nx)=-1; % Periodic B.C.'s
L = kron(L1,speye(nx,nx))+kron(speye(nx,nx),L1);
sqL = chol(L);

% Store necessary info for matrix vector multiply (A*x) function
a_fun              = 'Bmult_Dirichlet';
a_params.t_ext_hat = t_ext_hat;
a_params.L         = L;

% Construct Fourier representer for discrete Laplacian L for preconditioner.
l = zeros(2*nx,2*ny); 
l(1, 1) =  4; l(2 ,1) = -1; 
l(2*nx,1) = -1; l(1 ,2) = -1; 
l(1,2*ny) = -1; lhat = fft2(l);
clear l Ax PSF L1 Lsub Ldiag

% Store CG iteration information
params.max_cg_iter = 100;   
params.cg_step_tol = 1e-6;    
params.grad_tol = 1e-6;  
params.cg_io_flag = 0;  
params.cg_figure_no = [];
params.precond = 'Amult_Dirichlet';

%% MCMC sampling
nsamps  = 100;
nruns   = 2;
N       = nx*ny;
xsamp   = zeros(N,nsamps,nruns);
lamsamp = zeros(nsamps,nruns); lamsamp(1,:) = 7.5+rand(1,nruns);
delsamp = zeros(nsamps,nruns); delsamp(1,:) = 1e-4+0.01*rand(1,nruns);
tic
for i = 1:nsamps-1
    for k = 1:nruns
      h = waitbar(((i-1)*nruns+k)/(nsamps*nruns));
      %------------------------------------------------------------------
      % 1. Using conjugacy relationships, first sample the image. 
      temp = (lamsamp(i,k)*abs(t_ext_hat).^2+delsamp(i,k)*lhat);
      params.precond_params  = 1./temp;
      a_params.lambda        = lamsamp(i,k); 
      a_params.delta         = delsamp(i,k);
      params.a_params        = a_params;
      v                      = randn(nx,ny);
      rhs                    = Amult_Dirichlet(lamsamp(i,k)*b+sqrt(lamsamp(i,k))*v,conj(t_ext_hat))...
                                   + sqrt(delsamp(i,k))*reshape(sqL'*v(:),nx,ny);
      xtemp                  = CG(zeros(nx,ny),rhs,params,a_fun);
      xsamp(:,i,k)           = xtemp(:);
      %------------------------------------------------------------------
      % 2. Using conjugacy, sample the noise precision lam=1/sigma^2, 
      % conjugate prior: lam~Gamma(a0,1/t0), mean = a0/t0, var = a0/t0^2.
      a0=1; t0=0.0001; % uninformative prior values
      Axtemp = Amult_Dirichlet(xtemp,t_ext_hat);
      lamsamp(i+1,k) = gamrnd(a0+N/2,1./(t0+norm(Axtemp(:)-b(:))^2/2));
      %------------------------------------------------------------------
      % 3. Using conjugacy, sample regularization precisions delta, 
      % conjugate prior: delta~Gamma(a1,1/t1);
      a1=1; t1=0.0001; % uninformative prior values
      Lxtemp = reshape(L*xtemp(:),nx,ny);
      delsamp(i+1,k) = gamrnd(a1+N/2,1./(t1+xtemp(:)'*Lxtemp(:)/2));
    end
    %fprintf('%d samples out of %d in %3.2f seconds\n',(i-1)*nruns+k,nsamps*nruns,toc)
    %------------------------------------------------------------------
    % 4. Test mixing and convergence of the chains every nn samples
    % using technique on p. 296 of Gelman, et.al., Bayesian Data Analysis.
    nn = 50; clear Axtemp Lxtemp
    if mod(i,nn)==0 & nruns>1
        jj=floor(i/2)+1:i; ns = length(jj);
        xx = zeros(N+2,ns,nruns);
        xx(1:N,:,:)=xsamp(:,jj,:);
        xx(N+1,:,:)=lamsamp(jj,:);
        xx(N+2,:,:)=delsamp(jj,:);
        mean_j = sum(xx,2)/ns; % integrate each chain
        var_j = (1/(ns-1))*sum((xx-repmat(mean_j,[1,ns,1])).^2,2);
        clear xx jj
        mean_ij = sum(mean_j,3)/nruns; % integrate all samples
        B = ns/(nruns-1)*sum((mean_j-repmat(mean_ij,[1,1,nruns])).^2,3);
        W = sum(var_j,3)/nruns;
        Rhat = sqrt( ((ns-1)*W/ns+B/ns) ./ W );
        fprintf('Sample %d out of %d total; max Rhat = %2.3f.\n',i,nsamps,max(Rhat))
    end
end
toc
close(h)
%% Use last half of samples for analysis
xsamp = xsamp(:,floor(nsamps/2):nsamps-1,:); 
delsamp=delsamp(floor(nsamps/2):nsamps,:); 
lamsamp=lamsamp(floor(nsamps/2):nsamps,:);
qlam = quantile(lamsamp(:),[0.025,.975]);
fprintf('1/sigma^2 = %2.3f; 95 percent credibility interval: [%2.3f, %2.3f]\n',1/noise^2,qlam(1),qlam(2))
sampmean = mean(xsamp(:,:)');
relative_error = norm(sampmean(:)-x_true(:))/norm(x_true(:))
sampvar  = var(xsamp(:,:)');
figure(2), colormap(1-gray)
  imagesc(reshape(sampmean,nx,ny)), colorbar
figure(3), colormap(1-gray)
  imagesc(reshape(sqrt(sampvar),nx,ny)), colorbar
figure(4), colormap(1-gray)
  subplot(3,1,1), hist(delsamp(:),25), title('\delta, the prior precision')
  subplot(3,1,2), hist(lamsamp(:),25), title('\lambda, the noise precision')
  subplot(3,1,3), hist(delsamp(:)./lamsamp(:),25), title('\alpha=\delta/\lambda, the regularization parameter')
