function DAx = DA_mult(x,ahat)
Ax = real(ifft2(ahat.*fft2(x)));
[nx, ny] = size(x);
DAx = Ax(nx/4+1:3*nx/4,ny/4+1:3*ny/4);
