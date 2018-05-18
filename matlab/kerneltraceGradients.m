function [gParam, gX] = kerneltraceGradients(model, X)
  
if nargin < 2
  X = model.X;
end

g_meanFunc = [];
g_scaleBias = gpScaleBiasGradient(model);
gX = zeros(model.N, model.q);    
%%% Gradients of Kernel Parameters %%%
g_param = zeros(1, model.kern.nParams);
if isfield(model, 'beta')
  g_beta = 0;
else
  g_beta = [];
end    

%%% Compute Gradients with respect to X %%%
KX = kernCompute(model.kern, X, X);
model.invKX = pdinv(KX);
model.invKS = model.invKX * model.SX;

for i = 1:model.N          
  for j = 1:model.q
    gSX = SXGradXpoint(i,j,X);
    gKX = KXGradpoint(model.kern,i,j,X);
    temp = -(gKX*model.invKS) + gSX;
    gX(i, j) = gX(i, j) - model.mu * trace(model.invKX *temp);           
  end
end
%%% Compute Gradients of Kernel Parameters %%%
gK = localCovarianceGradients(model);
g_param = g_param + kernGradient(model.kern, X, gK);
      
if nargout < 4
   if (~isfield(model, 'optimiseBeta') && ~strcmp(model.approx, 'ftc')) ...
          | model.optimiseBeta
      % append beta gradient to end of parameters
      gParam = [g_param(:)' g_meanFunc g_scaleBias g_beta];
   else
      gParam = [g_param(:)' g_meanFunc g_scaleBias];
   end
else
    gParam = [g_param(:)' g_meanFunc g_scaleBias];
end  
end
 
function gK = localCovarianceGradients(model)
gK = model.mu*model.invKX*model.SX*model.invKX;
end  

function gSX = SXGradXpoint(i,j, X)
gSX =zeros(size(X,1));
x = X(i,:);
n2 = dist2(X,x);
wi2 = 0.5 ;
rbfPart = exp(-n2*wi2);
for k = 1: size(X,1) 
    if (k == i) 
        gSX(:, k) = (X(:, j) - x(j)).*rbfPart;
        gSX(k, :) = gSX(:, k)';
    end
end
end

%rbf kernel
function gKX = KXGradpoint(kern,i,j,X)
gKX = zeros(size(X,1));
x = X(i,:);
n2 = dist2(X, x);
wi2 = (0.5 * kern.comp{1}.inverseWidth);
rbfPart = kern.comp{1}.variance*exp(-n2*wi2);
for k = 1: size(X,1) 
    if (k == i) 
       gKX(:, i) = kern.comp{1}.inverseWidth*(X(:, j) - x(j)).*rbfPart;
       gKX(k, :) = gKX(:, k)';
    end
end
if isfield(kern, 'isNormalised') && (kern.isNormalised == true)
    gKX = gKX * sqrt(kern.inverseWidth/(2*pi));
end
end

