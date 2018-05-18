function [E,v,Xp] = km_pca(X,m)
% KM_PCA calculates the principal directions and principal components of a 
% data set X.
% Input:	- X: data matrix in row format (each data point is a row)
%			- m: the number of principal components to return. If m is 
%			smaller than 1, it is interpreted as the fraction of the signal
%			energy that is to be contained within the returned principal
%			components.
% Output:	- E: matrix containing the principal directions.
%			- v: array containing the eigenvalues.
%			- Xp: matrix containing the principal components
% USAGE: [E,v,Xp] = km_pca(X,m)
%
% Author: Steven Van Vaerenbergh (steven *at* gtas.dicom.unican.es), 2010.
% Id: km_pca.m v1.0
% This file is part of the Kernel Methods Toolbox (KMBOX) for MATLAB.
% http://sourceforge.net/p/kmbox
%
% This program is free software: you can redistribute it and/or modify it 
% under the terms of the GNU General Public License as published by the 
% Free Software Foundation, version 3 (as included and available at
% http://www.gnu.org/licenses).

N = size(X,1);
[E,V] = eig(X'*X/N);

v = diag(V);
[v,ind] = sort(v,'descend');
E = E(:,ind);

Xp = X*E(:,1:m);