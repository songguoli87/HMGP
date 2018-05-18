function [gParam, gX_u, gX] = FastkernelFconstraintGradients(model, X, X_u)
  
  if nargin < 3
    if isfield(model, 'X_u')
      X_u = model.X_u;
    else
      X_u = [];
    end
    if nargin < 2
      X = model.X;
    end
  end

  gX_u = [];
  gX = [];
  
  g_meanFunc = [];
  g_scaleBias = [];  
  
      %%% Gradients of Kernel Parameters %%%
%     g_param = zeros(1, model.kern.nParams);
    if isfield(model, 'beta')
      g_beta = 0;
    else
      g_beta = [];
    end

switch model.approx
   case 'ftc'   
    % Full training conditional.       
    %%% Gradients of Kernel Parameters %%%
    g_param = zeros(1, model.kern.nParams);
    if isfield(model, 'beta')
      g_beta = 0;
    else
      g_beta = [];
    end        
    gK = localCovarianceGradients(model);
    %%% Compute Gradients with respect to X %%%
      for i = 1:model.N          
        for j = 1:model.q
            gSX = SXGradXpoint(i,j,X);
            gKX = KXGradpoint(model.kern,i,j,X);
           gX(i, j) = gX(i, j) + sum(sum((gKX - gSX).*gK)); % F norm
        end
      end
    %%% Compute Gradients of Kernel Parameters %%%
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
  
  case {'dtc', 'dtcvar', 'fitc', 'pitc'}
   % Sparse approximations.
    gSKuu = localConstraintGradients(model);       
    K_uuInvK_uf = model.invK_uu*model.K_uf;        
    gK_u = -2* K_uuInvK_uf * gSKuu * K_uuInvK_uf';    
    gK_uf = 4*K_uuInvK_uf *gSKuu;
          
    %%% Compute Gradients of Kernel Parameters %%%
    gParam_u = kernGradient(model.kern, X_u, gK_u);
    gParam_uf = kernGradient(model.kern, X_u, X, gK_uf);

    g_param = gParam_u + gParam_uf;      
    
    %%% Compute Gradients with respect to X_u %%%
    gKX = kernGradX(model.kern, X_u, X_u);
    % The 2 accounts for the fact that covGrad is symmetric
    gKX = gKX*2;
    
    dgKX = kernDiagGradX(model.kern, X_u);
    for i = 1:model.k
      gKX(i, :, i) = dgKX(i, :);
    end
    
    if ~model.fixInducing | nargout > 1
      % Allocate space for gX_u
      gX_u = zeros(model.k, model.q);
      % Compute portion associated with gK_u
        for i = 1:model.k
          for j = 1:model.q
            gX_u(i, j) = gKX(:, j, i)'*gK_u(:, i);
          end
        end

      % Compute portion associated with gK_uf
      gKX_uf = kernGradX(model.kern, X_u, X);
      for i = 1:model.k
        for j = 1:model.q
          gX_u(i, j) = gX_u(i, j) + gKX_uf(:, j, i)'*gK_uf(i, :)';
        end
      end
    end
    if nargout > 2
      %%% Compute gradients with respect to X %%%      
      % Allocate space for gX
      gXk = zeros(model.N, model.q);      
      % this needs to be recomputed so that it is wrt X not X_u
      gKX_uf = kernGradX(model.kern, X, X_u);      
      for i = 1:model.N
        for j = 1:model.q
          gXk(i, j) = gKX_uf(:, j, i)'*gK_uf(:, i);
        end
      end    
    end
    
    gXs = zeros(model.N, model.q);
       gS_x = -2*gSKuu;       
       for i = 1:model.N          
            for j = 1:model.q
                gSX = SXGradXpoint(i,j,X);
                gXs(i, j) = gXs(i, j) + sum(sum(gSX.*gS_x));
            end
       end
        gX = gXk + gXs;
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
 
function gK = localCovarianceGradients(model)
K = kernCompute(model.kern, model.X, model.X);
gK = -model.mu*(K - model.SX); % F norm
end  

function gSKuu = localConstraintGradients(model)
K = model.K_uf'*model.invK_uu*model.K_uf;
gSKuu = -model.mu*(K - model.SX); % F norm
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


