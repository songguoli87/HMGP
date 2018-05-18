function [gParam, gX_u] = FastkerneltraceGradients_xu(model, X_u)
  
  if nargin < 2
    if isfield(model, 'X_u')
      X_u = model.X_u;
    else
      X_u = [];
    end
  end

 g_meanFunc = [];
 g_scaleBias = [];
 if isfield(model, 'beta')
    g_beta = 0;
 else
    g_beta = [];
 end
 
 gX_u = zeros(model.k, model.q);
   
 switch model.approx
   case 'fitc'   
    Suu = localsimilarity(X_u); 
    model.invKS = model.invK_uu * Suu;    
    %%% Compute Gradients with respect to X %%%
      for i = 1:model.k          
        for j = 1:model.q
            gSXu = SXGradXpoint(i,j,X_u);
            gKXu = KXGradpoint(model.kern,i,j,X_u);
            temp = -(gKXu*model.invKS) + gSXu;
            gX_u(i, j) = gX_u(i, j) - model.mu * trace(model.invK_uu *temp);
        end
      end
   %%% Compute Gradients of Kernel Parameters %%%
    gKu = localCovarianceGradients(model,Suu);
    g_param = kernGradient(model.kern, X_u, gKu); 
  otherwise
    error('Unknown model approximation.')
 end 
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

function Suu = localsimilarity(X_u) 
    X = X_u;
    nsq=sum(X.^2,2);
    K=bsxfun(@minus,nsq,(2*X)*X.');
    K=bsxfun(@plus,nsq.',K);
    wi2=  0.5;
    Suu = exp(-K*wi2);
end

function gK = localCovarianceGradients(model,Suu)
gK = -model.mu*model.invK_uu*Suu*model.invK_uu;
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


