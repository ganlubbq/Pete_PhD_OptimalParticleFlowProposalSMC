function [ Erfx ] = erfm( X )
%ERFM Matrix error function

f_erfderiv = @nderiv;
Erfx = funm(X, f_erfderiv);

end

function dn = nderiv(x, N)
% Returns the Nth derivative of the error function.

if N == 0
    dn = erf(x);
    
else
    % Hermite Polynomial
    n = N - 1;
    Hn = 1;
    for kk = 1:floor(n/2)
        Hn = Hn + (-1)^kk * (2*x).^(n-2*kk) / (factorial(kk)*factorial(n-2*kk));
    end
    Hn = Hn * factorial(n);
    
    dn = (-1)^n * Hn .* exp(-x.^2);
    
end


end