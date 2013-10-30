  function b = Amult_Dirichlet(x,a_hat)
%
%  Compute b = A*x for A embedded in a periodic matrix.
  [Nx,Ny] = size(a_hat);
  [Nxd2,Nyd2] = size(x);
  if (Nx/2 ~= Nxd2 | Ny/2 ~= Nyd2)
    fprintf('*** Row and col dim of 1st arg must be half that of 2nd arg.\n');
    return
  end

  %  Extend x by zeros.
  x_extend = zeros(Nx,Ny);
  x_extend(1:Nxd2,1:Nyd2) = x;

  %  Compute product using FFTs; then restrict.
  b_extend = real(ifft2( a_hat .* fft2(x_extend) ));
  b = b_extend(1:Nxd2,1:Nyd2);


