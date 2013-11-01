function AtDtx = AtDt_mult(x,ahat)
[nnx, nny] = size(x);
Dtx = padarray(x,[nnx/2,nny/2]);
AtDtx = real(ifft2(conj(ahat).*fft2(Dtx))); 
