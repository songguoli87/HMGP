function [Xcca,Ycca,A, B,test] = cca3(X,Y,Xte,Yte)
%CCA3 computes canonical correlation matrices
%   Returns the rotation matrices A and B for modalities
%   X and Y respectively. As well as the CCA version of 
%   X and Y, Xcca and Ycca respectively.
%
%   [Xcca,Ycca, A, B, test] = cca3(X,Y,Xte,Yte)
%
%   If test rbfples of X and Y are provided, it also 
%   returns their cca version in the struct test:
%   	test.Xcca
%   	test.Ycca

   % addpath('D:\SGL\experiments\KMBOX-0.9\KMBOX-0.9\');
%     addpath('E:\experiments\CCA-kCCA\CCA-kCCA');
%     vl_setup('noprefix');
%       Mmax = 100;  % max. M (number of components in incomplete Cholesky decomp.)
%        reg = 1E-5; % regularization
%     kerneltype1 = 'gauss';   % kernel type
%     kerneltype = 'gauss';
%     kernelpar = 1;  % kernel parameter
%     eta=0.01;
%     kapa=1;
    
    
    vX = sqrt(var(X,1));
    vY = sqrt(var(Y,1));
    mX = mean(X,1);
    mY = mean(Y,1);
    
    X = (X - repmat(mX,size(X,1),1))./repmat(vX,size(X,1),1);
    Y = (Y - repmat(mY,size(Y,1),1))./repmat(vY,size(Y,1),1);
    
    X(find(isnan(X))) = 0;
    Y(find(isnan(Y))) = 0;       
   
%      [Xcca,Ycca,A,B,beta] = km_kcca(X,Y,kerneltype,kernelpar,reg,'full');
%     Kx = km_kernel(X,X,kerneltype,kernelpar);
%     Ky = km_kernel(Y,Y,kerneltype,kernelpar);    
% 
%     [nalpha, nbeta, r, Kx_c, Ky_c] =  kcanonca_reg_ver2(Kx,Ky,eta,kapa);
%      Xcca = Kx_c*nalpha;
%      Ycca = Ky_c*nbeta;
    

    [A,B] = canoncorr(X,Y);
     Xcca = X*A;
     Ycca = Y*B;
  
    if nargin>2,
	assert(size(Xte,1) == size(Yte,1),'X_test and Y_test have different number of rbfples');
 	n=size(Xte,1);
   
 	scaled_Xte = ((Xte - repmat(mX,n,1)))./repmat(vX,n,1);
 	scaled_Xte(isnan(scaled_Xte)) = 0;
  	test.Xcca = scaled_Xte * A;

 	scaled_Yte = ((Yte - repmat(mY,n,1)))./repmat(vY,n,1);
 	scaled_Yte(isnan(scaled_Yte)) = 0;
  	test.Ycca = scaled_Yte * B;
      
%     Kxte = km_kernel(scaled_Xte,X,kerneltype,kernelpar);
%     Kyte = km_kernel(scaled_Yte,Y,kerneltype,kernelpar);
    
%     GX = GX-repmat(mean(GX),n,1);
% 	GY = GY-repmat(mean(GY),n,1);
    
%     Kxte_c = kernelcenter(Kx,Kxte);   
%     Kyte_c = kernelcenter(Ky,Kyte);
%     test.Xcca = Kxte*A;
%     test.Ycca = Kyte*B;

    else
	test=0;
    end;

