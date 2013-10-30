  function y = bttb_mult_fft(x,t_hat)

%
%  Compute y = BTTB(t)*x using block circulant extension and 2-D FFTs.
%  t_hat = fft2(t).

  [Nx,Ny] = size(t_hat);
  [Nxd2,Nyd2] = size(x);
  if (Nx/2 ~= Nxd2 | Ny/2 ~= Nyd2)
    fprintf('*** Row and col dim of 1st arg must be half that of 2nd arg.\n');
    return
  end

  %  Extend x by zeros.
  
  x_extend = zeros(Nx,Ny);
  x_extend(1:Nxd2,1:Nyd2) = x;

  %  Compute product using FFTs; then restrict.
  
  y_extend = real(ifft2( t_hat .* fft2(x_extend) ));
  y = y_extend(1:Nxd2,1:Nyd2);


