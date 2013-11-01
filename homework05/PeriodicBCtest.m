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
  h = 1/nx;
  x = [-0.5+h/2:h:0.5-h/2]';
  [X,Y]=meshgrid(x);
  sig = 2*h;
  kernel = exp(-((X-h/2).^2+(Y-h/2).^2)/2/sig^2);
  kernel = kernel/sum(sum(kernel));
  Ax = real(ifft2(fft2(fftshift(kernel)).*fft2(x_true)));
  Ax=Ax(101:228,101:228);
  x_true = x_true(101:228,101:228);
  [nx,ny] = size(Ax);
  err_lev = 2;
  noise = err_lev/100 * norm(Ax(:)) / sqrt(nx*ny);%input(' The standard deviation of the noise = ');
  b = Ax + noise*randn(nx,ny);
  %figure(1), imagesc(x_true), colormap(gray), colorbar 
  %figure(2), imagesc(b,[0,max(x_true(:))]), colormap(gray), colorbar 
  
  % Use 128^2 PSF model with periodic BCs and GCV choice of alpha
  kernel2 = kernel(45:212,45:212);
  kernel = kernel(65:192,65:192); 

n = nx*ny;
ahat = fft2(fftshift(kernel));
bhat = fft2(b);
G_fn=@(a)(sum(sum((a^2*abs(bhat).^2)./(abs(ahat).^2+a).^2))) / ...
	 (n-sum(sum(abs(ahat).^2./(abs(ahat).^2+a))))^2;
gcv_alpha =  fminbnd(G_fn,0,1);

xalpha = @(a) real(ifft2((conj(ahat)./(abs(ahat).^2+a).*bhat)));
aa = logspace(-8,0);
figure()
  subplot(2,2,1), imagesc(x_true), colorbar, colormap(gray), title('True Image')
  subplot(2,2,2), imagesc(b), colorbar, colormap(gray), title('Blurred Noisy Image')
  subplot(2,2,3), imagesc(xalpha(gcv_alpha),[0,max(x_true(:))]), colorbar, colormap(gray), title(sprintf('GCV Construction: alpha = %2.3e',gcv_alpha))
  subplot(2,2,4), loglog(aa,arrayfun(G_fn,aa),'b-', gcv_alpha,G_fn(gcv_alpha),'b*'), xlim([1e-8,1]), title('GCV Curve')
set(gcf,'PaperPosition', [0 0 10 8])
set(gcf,'PaperSize', [10 8]) 
saveas(gcf,'boundary_bad.pdf') 

% Stupid idea 
%[nrow, ~] = size(b);
%pad = zeros(nrow,20); 
%b_pad = [pad b pad]; 
%[~, ncol] = size(b_pad);
%pad = zeros(20,ncol);
%b_pad = [pad; b_pad; pad];
%[nx, ny] = size(b_pad);
%n = nx*ny;
%ahat = fft2(fftshift(kernel2));
%bhat = fft2(b_pad);
%G_fn=@(a)(sum(sum((a^2*abs(bhat).^2)./(abs(ahat).^2+a).^2))) / ...
%	 (n-sum(sum(abs(ahat).^2./(abs(ahat).^2+a))))^2;
%gcv_alpha =  fminbnd(G_fn,0,1);
%
%xalpha = @(a) real(ifft2((conj(ahat)./(abs(ahat).^2+a).*bhat)));
%  subplot(2,2,4), imagesc(xalpha(gcv_alpha)), colorbar, colormap(gray)

