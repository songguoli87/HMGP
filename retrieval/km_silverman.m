function kernel_width = km_silverman(x)
% KM_SILVERMAN uses Silverman's rule to determine the kernel width for 
% performing kernel density estimation with a Gaussian kernel.
%
% Input:	- x: array of 1D data
% Output:	- kernel_width
% USAGE: kernel_width = km_silverman(x)
%
% Author: Steven Van Vaerenbergh (steven *at* gtas.dicom.unican.es), 2011.
% Id: km_silverman.m v1.0
% This file is part of the Kernel Methods Toolbox (KMBOX) for MATLAB.
% http://sourceforge.net/p/kmbox
%
% The code in this file is based on the following publication:
% B. W. Silverman. Density estimation for Statistics and Data Analysis.
% Chapman & Hall/CRC, London, UK, 1986.
%
% This program is free software: you can redistribute it and/or modify it
% under the terms of the GNU General Public License as published by the
% Free Software Foundation, version 3 (http://www.gnu.org/licenses).

sig = std(x);
sx = sort(x);
q3 = median(sx(ceil(length(sx)/2):end));
q1 = median(sx(1:floor(length(sx)/2)));
kernel_width = 0.9*min(sig,(q3-q1)/1.34)*length(x)^-0.2;
