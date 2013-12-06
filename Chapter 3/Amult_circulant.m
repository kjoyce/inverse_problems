function Ax = Afun_circulant(x,ahat)
Ax = real(ifft2(ahat.*fft2(x)));
