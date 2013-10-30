function AtDtx = AtDt_mult(x,ahat)
[nx, ny] = size(x);
Dtx = padarray(x,[nx/2,ny/2]);
ADtx = real(ifft(ahat.*fft2(Dtx))); 
AtDtx = ADtx; % Recall A is symmetric
