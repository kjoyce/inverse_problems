function[Dalpha] = DP_fn(alpha,b,c,params,Bmult_fn,sigma)

a_params               = params.a_params;
a_s_hat                = a_params.a_s_hat;
[nx2,ny2]              = size(a_s_hat); nx = nx2/2; ny = ny2/2; 
a_params.alpha         = alpha;
params.a_params        = a_params;
params.precond_params  = 1./(abs(a_s_hat).^2+alpha);
xalpha                 = CG(zeros(nx,ny),c,params,Bmult_fn);
Axalpha                = Amult_Dirichlet(xalpha,a_s_hat);

% Evaluate DP function.
Dalpha                 = norm(Axalpha(:)-b(:))^2-nx*ny*sigma^2;