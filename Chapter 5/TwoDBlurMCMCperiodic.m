%  
%  Image deblurring inverse problem with periodic boundary conditions.
%
%% Generate data and build regularization matrix.
clear all, close all
load satellite
%x_true = x_true/max(x_true(:));
[nx,ny] = size(x_true); 
N = nx*ny;
h = 1/nx;
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
eta =  sigma * randn(nx,ny);
b = Ax + eta;
bhat = fft2(b);
figure(1)
  imagesc(x_true), colorbar, colormap(1-gray)
figure(2)
  imagesc(b), colorbar, colormap(1-gray)

% Construct Fourier representer for discrete Laplacian L.
l = zeros(nx,ny);
l(1, 1) =  4; l(2 ,1) = -1;
l(nx,1) = -1; l(1 ,2) = -1;
l(1,ny) = -1; lhat = real(fft2(l));
clear l Ax PSF

%% MCMC sampling
nsamps = 500;
nruns = 5;
Rhat_tol = 1.0;
xsamp = zeros(N,nsamps,nruns);
lamsamp = zeros(nsamps,nruns); lamsamp(1,:) = 5+5*rand(1,nruns);
delsamp = zeros(nsamps,nruns); delsamp(1,:) = 0.5*rand(1,nruns);
i = 0; iter_flag = 1;
tic
while iter_flag
  i = i+1;
  if i == nsamps-1, iter_flag = 0; end
  %fprintf('iteration = %d\n',i)
  for k = 1:nruns
      h = waitbar(((i-1)*nruns+k)/(nsamps*nruns));
      %------------------------------------------------------------------
      % 1. Using conjugacy relationships, first sample the image. 
      fourier_filt  = lamsamp(i,k)*abs(ahat).^2 + delsamp(i,k)*lhat;
      xtemp = real(ifft2(conj(ahat).*(lamsamp(i,k)*bhat)./fourier_filt + ...
                         fft2(randn(nx,ny))./sqrt(fourier_filt)));
      xsamp(:,i,k) = xtemp(:);
      %------------------------------------------------------------------
      % 2. Using conjugacy, sample the noise precision lam=1/sigma^2, 
      % conjugate prior: lam~Gamma(a0,1/t0), mean = a0/t0, var = a0/t0^2.
      a0=1; t0=0.0001; % uninformative prior values
      Axtemp = real(ifft2(ahat.*fft2(xtemp)));
      lamsamp(i+1,k) = gamrnd(a0+N/2,1./(t0+norm(Axtemp(:)-b(:))^2/2));
      %------------------------------------------------------------------
      % 3. Using conjugacy, sample regularization precisions delta, 
      % conjugate prior: delta~Gamma(a1,1/t1);
      a1=1; t1=0.0001; % uninformative prior values
      Lxtemp = real(ifft2(lhat.*fft2(xtemp)));
      delsamp(i+1,k) = gamrnd(a1+(N-1)/2,1./(t1+xtemp(:)'*Lxtemp(:)/2));
    end
    %------------------------------------------------------------------
    % 4. Test mixing and convergence of the chains every nn samples
    % using technique on p. 296 of Gelman, et.al., Bayesian Data Analysis.
    nn = 100; clear Axtemp Lxtemp 
    if mod(i+1,nn)==0 & nruns>1
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
      if Rhat < Rhat_tol, iter_flag = 0; end
    end
end
toc
close(h), clear B W    
%% Use last half of samples for analysis
xsamp = xsamp(:,floor(i/2):i,:); 
delsamp=delsamp(floor(i/2):i+1,:); 
lamsamp=lamsamp(floor(i/2):i+1,:);
qlam = quantile(lamsamp(:),[0.025,.975]);
fprintf('1/sigma^2 = %2.3f; 95 percent credibility interval: [%2.3f, %2.3f]\n',1/sigma^2,qlam(1),qlam(2))
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
% figure(5), imagesc(b-real(ifft2(ahat.*fft2(reshape(sampmean,nx,ny))))), colormap(1-gray), colorbar