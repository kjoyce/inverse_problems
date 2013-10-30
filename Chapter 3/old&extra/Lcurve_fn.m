function[Calpha] = Lcurve_fn(alpha,b,c,params,Bmult_fn)

a_params               = params.a_params;
a_s_hat                = a_params.a_s_hat;
[nx2,ny2]              = size(a_s_hat); nx = nx2/2; ny = ny2/2; 
a_params.alpha         = alpha;
params.a_params        = a_params;
params.precond_params  = 1./(abs(a_s_hat).^2+alpha);
xalpha                 = CG(zeros(nx,ny),c,params,Bmult_fn);
Axalpha                = Amult_Dirichlet(xalpha,a_s_hat);
c_new                  = Amult_Dirichlet(Axalpha-b,conj(a_s_hat));
zalpha                 = CG(zeros(nx,ny),c_new,params,Bmult_fn);
s                      = norm(xalpha)^2;
r                      = norm(Axalpha - b)^2;
sp                     = 2/alpha*xalpha(:)'*zalpha(:);

% Evaluate DP function.
Calpha                 = (r*s*(alpha*r + alpha^2*s) + (r*s)^2/sp)/(r^2+alpha^2*s^2)^(3/2);