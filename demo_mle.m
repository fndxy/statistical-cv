% This is a demo for segmentation using local gaussian distribution (LGD)
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


clc;clear all;close all;

Img=imread('three.bmp');
% Img = double(Img(:,:,1));
origin = Img;
%  origin = imresize(origin,0.25);
Img = double(rgb2gray(Img));
%  Img = imresize(Img,0.25);
 
[m,n] = size(Img);


 
NumIter = 200; %iterations
timestep=0.2; %time step
mu=0.1/timestep;% level set regularization term, please refer to "Chunming Li and et al. Level Set Evolution Without Re-initialization: A New Variational Formulation, CVPR 2005"
sigma = 5;%size of kernel
epsilon = 1;
c0 = 2; % the constant value 
lambda1=1.05;%outer weight, please refer to "Chunming Li and et al,  Minimization of Region-Scalable Fitting Energy for Image Segmentation, IEEE Trans. Image Processing, vol. 17 (10), pp. 1940-1949, 2008"
lambda2=1.0;%inner weight
%if lambda1>lambda2; tend to inflate
%if lambda1<lambda2; tend to deflate
nu = 0.001*255*255;%length term
alf = 30;%data term weight


figure,imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal
 [Height Wide] = size(Img);
 [xx yy] = meshgrid(1:Wide,1:Height);
 phi = (sqrt(((xx - 64).^2 + (yy - 65).^2 )) - 17);
 phi = sign(phi).*c0;

 [nrow,ncol] =size(Img);
 figure(1);
 imshow(Img)
 ic=nrow/2;
 jc=ncol/2;
 r=ic*0.55;
 
  %phi_0 = sdf2circle(nrow,ncol,ic,jc,r);
  [X,Y] = meshgrid(1:ncol, 1:nrow);
  phi = sqrt((X-jc).^2+(Y-ic).^2)-r;  %��ʼ��phiΪSDF
    r = ic/15;
    tm= 4;
    [m,n] = size(Img(:,:,1));
    phi = -2*ones(m,n);
    for iter1 = 0:tm*2;
        for iter2 = 0:tm*2;
        center_x = iter1*ic/tm;
        center_y = iter2*jc/tm;
        for i = 1:m
           for j = 1:n
               ttt = (i-center_x)^2 + (j-center_y)^2;
               if (sqrt(ttt) - r)<0
                   phi(i,j) = 2;
              end
          end
        end
        end
    end
% 
% 
  [row,col] = size(Img);
   phi = ones(row,col);
   phi(3:row-3,3:col-3) = -1;
   phi = - phi;
   [c, h] = contour(phi, [0 0], 'r');
   title('Initial contour');


figure(2),imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal,
hold on,[c,h] = contour(phi,[0 0],'r','linewidth',1); hold off
pause(0.5)

tic
for iter = 1:200
    phi =maxlikihood(Img,phi,epsilon,mu,nu,timestep,alf);

   if(mod(iter,50) == 0)
         figure(3),
         imagesc(uint8(Img),[0 255]),colormap(gray),axis off;axis equal,title(num2str(iter))
         hold on,[c,h] = contour(phi,[0 0],'r','linewidth',1); hold off
         pause(0.1);
     end

end


toc
