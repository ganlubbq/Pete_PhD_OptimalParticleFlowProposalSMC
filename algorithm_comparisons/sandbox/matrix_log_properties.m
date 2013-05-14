n = 5;

r = 3;
z = mvnrnd(zeros(1,n), eye(n), r)';
X = z*z';
I = eye(n);
t = rand;
A = -0.5*(I+t*X)\X;
P = -0.5*logm(I+t*X)*X;
A*P-P*A

A = -0.5*(I+t*X)\X;
P = 0.5*logm(I+t*X)+0.5*(I+t*X);
A*P-P*A

r = 5;
z = mvnrnd(zeros(1,n), eye(n), r)';
X = z*z';
sqrtm(inv(X))-inv(sqrtm(X))

I = eye(n);
A = X;
B = logm(X);
O = zeros(n);
sqrtm([(I+A) O; B I])-[sqrtm(I+A) O; B*(A\(sqrtm(I+A)-I)) I]

sqrtm(X)*X-X*sqrtm(X)

sqrt(det(X))-det(sqrtm(X))

sqrtm(A)*sqrtm(B) - sqrtm(A*B)

%%
clup
ds = 5;
do = 5;
Q = 0.9*eye(ds)+0.1*ones(ds);
R = 0.5*eye(do)+0.5*ones(do);
H = eye(do);
% R = 0.01*eye(do)+0.99*ones(do);
% H = [1 1 0 0 0; 0 0 1 1 0; 0 0 0 0 1];
[W, Omega] = eig(Q*H'*(R\H));
XXT = (W\Q)/W';
C = zeros(ds);
for ii = 1:ds
    for jj = 1:ds
        C(ii,jj) = ( exp(1+0.5*(Omega(ii,ii)+Omega(jj,jj))) - exp(1))/(1+0.5*(Omega(ii,ii)+Omega(jj,jj)));
    end
end
Itgrl = W*(XXT.*C)
eig(Itgrl)