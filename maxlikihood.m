% This is the evolution step for segmentation using local gaussian distribution (LGD)
% fitting energy
%
% Reference: <Li Wang, Lei He, Arabinda Mishra, Chunming Li. 
% Active Contours Driven by Local Gaussian Distribution Fitting Energy.
% Signal Processing, 89(12), 2009,p. 2435-2447>
%
% Please DO NOT distribute this code to anybody.
% Copyright (c) by Li Wang
%
% Author:       Li Wang
% E-mail:       li_wang@med.unc.edu
% URL:          http://www.unc.edu/~liwa/
%
% 2010-01-02 PM

function [u ]= maxlikihood(Img,u,epsilon,mu,nu,timestep,alf)

u=NeumannBound(u);
K=curvature_central(u); 
H=Heaviside(u,epsilon);
Delta = Dirac(u,epsilon);




I = Img;
u1 = sum(sum(H.*I))/sum(sum(H));
u2 = sum(sum((1-H).*I))/sum(sum(1-H));







sigma1 = sum(sum((I - u1).^2.*H))/sum(sum(H));
sigma2 = sum(sum((I - u1).^2.*(1-H)))/sum(sum(1-H));
sigma1 = sigma1 + eps;
sigma2 = sigma2 + eps;



sub1 = (u1 - Img).^2;
sub2 = (u2 - Img).^2;
e1 = log(sqrt(sigma1))+(sub1/sigma1);
e2 = log(sqrt(sigma2))+(sub2/sigma2);
localForce = e1 - e2;

A = -alf.*Delta.*localForce;%data force
P=mu*(4*del2(u) - K);% level set regularization term, please refer to "Chunming Li and et al. Level Set Evolution Without Re-initialization: A New Variational Formulation, CVPR 2005"
L=nu.*Delta.*K;%length term



[f_fx,f_fy]=forward_gradient(Img);
grad_im = (f_fx.^2 + f_fy.^2 + 1e-10).^0.5;
kk=1;                        % 反差参数
g=1./(1+(grad_im/kk).^2);    % 计算边缘函数g


ch=4;
 [K , s]=curtative_g(u,g);
 temp1 = ch*s;
 
D = s.*K + g.*temp1;
u = u+timestep*(L+P+A);

return;



function g = NeumannBound(f)
[nrow,ncol] = size(f);
g = f;
g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);
g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);
g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);


function K = curvature_central(u);
[bdx,bdy]=gradient(u);
mag_bg=sqrt(bdx.^2+bdy.^2)+1e-10;
nx=bdx./mag_bg;
ny=bdy./mag_bg;
[nxx,nxy]=gradient(nx);
[nyx,nyy]=gradient(ny);
K=nxx+nyy;


function h = Heaviside(x,epsilon)
h=0.5*(1+(2/pi)*atan(x./epsilon));

function f = Dirac(x, epsilon)
f=(epsilon/pi)./(epsilon^2.+x.^2);


