function[Ax]=Amult(x,ahat)

% Compute Ax using ffts.
Ax = real(ifft2(ahat.*fft2(x)));
