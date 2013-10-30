  function y_array = Bmult_Dirichlet(x_array,params);

%
%
%  Compute array(y), where y = (T'*T + alpha*L)*x. T is assumed to be
%  block Toeplitz with Toeplitz blocks (BTTB). Block circulant
%  extensions are used to compute matrix-vector products involving T
%  and T'. The products have O(n log n) complexity through the use of
%  2-D FFT's. 

  a_s_hat    = params.a_s_hat;
  alpha      = params.alpha;
  TtTx_array = Amult_Dirichlet(Amult_Dirichlet(x_array,a_s_hat),conj(a_s_hat));
  y_array    = TtTx_array + alpha*x_array;
  
  