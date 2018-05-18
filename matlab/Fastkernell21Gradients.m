function [gParam, gX_u, gX] = Fastkernell21Gradients(model, X, X_u)
  
  if nargin < 4
    if isfield(model, 'X_u')
      X_u = model.X_u;
    else
      X_u = [];
    end
    if nargin < 3 && ~isfield(model, 'S')
      M = model.m;
    end
    if nargin < 2
      X = model.X;
    end
  end

  gX_u = [];
%   gX = [];
  gX = zeros(model.N,model.q);
  
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
        model.invKS = model.invK_uu * model.SX;
          %%% Compute Gradients with respect to X %%%
          for i = 1:model.N          
            for j = 1:model.q
                gSX = SXGradXpoint(i,j,X);
                gSX = gSX*2;
                gKX = KXGradpoint(model.kern,i,j,X);
                gKX = gKX*2;
                temp = -(gKX*model.invKS) + gSX;
             gX(i, j) = gX(i, j) - model.mu * trace(model.invK_uu *temp);
            end
          end
          %%% Compute Gradients of Kernel Parameters %%%
%           gX = gX - gK*X ;
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
  
  case {'dtc', 'dtcvar', 'fitc', 'pitc'}
       % Sparse approximations.
%         gSKuu = localConstraintGradients(model);
    
%     invK_uuK_uf = model.invK_uu*model.K_uf;
%     gK_u = model.mu * invKSuu * invKuu;
     
%     K_uuK_uf2 = K_uuInvK_uf*K_uuInvK_uf';

%     K_uf2 = model.K_uf*model.K_uf';
%     model.A = (1/model.beta)*model.K_uu+ K_uf2;
%     % This can become unstable when K_uf2 is low rank.
%     [model.Ainv, U] = pdinv(model.A);
    
%     E = model.K_uf*model.Dinv;  
%     ES = E * model.SX;
    E = model.K_uf*model.Dinv*X;
%     EET = E*E';
    AinvE = model.Ainv*E;
%     AinvEETAinv = AinvE*model.SX*AinvE';
    AinvEETAinv = AinvE*AinvE';
    diagK_ufdAinvplusAinvEETAinvK_fu = ...
        sum(model.K_uf.*((model.d*model.Ainv+model.beta*AinvEETAinv)*model.K_uf), 1)';
    invK_uuK_uf = model.invK_uu*model.K_uf;
    invK_uuK_ufDinv = invK_uuK_uf*model.Dinv;    
%     diagMMT = sum(model.SX, 2);
    diagMMT = sum(X.*X, 2);
%     diagK_fuAinvEMT = sum(model.K_uf.*(model.Ainv*ES), 1)';
    diagK_fuAinvEMT = sum(model.K_uf.*(model.Ainv*E*X'), 1)';
    diagQ = -model.d*model.diagD + model.beta*diagMMT ...
            + diagK_ufdAinvplusAinvEETAinvK_fu...
            -2*model.beta*diagK_fuAinvEMT;
    gK_u = model.mu *(- AinvEETAinv ...
                 + model.beta*invK_uuK_ufDinv*sparseDiag(diagQ)*invK_uuK_ufDinv');
%     gK_uf = 2*model.mu * (-model.beta*AinvEETAinv*model.K_uf*model.Dinv ...
%             + model.beta*(model.Ainv*ES*model.Dinv));
    gK_uf = 2*model.mu * (-model.beta*AinvEETAinv*model.K_uf*model.Dinv ...
            + model.beta*(model.Ainv*E*X'*model.Dinv));
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
          gX_u(i, j) =  gKX(:, j, i)'*gK_u(:, i);
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
       
    gXs = zeros(model.N, model.q);    
%     K = kernCompute(model.kern, X);
%     invK = pdinv(K);
%     Dinvm = model.Dinv*model.SX;
%     K_ufDinvm = model.K_uf*model.Dinv;    
        Dinvm = model.Dinv*X;
        K_ufDinvm = model.K_uf*Dinvm;
     for i = 1:model.N          
            for j = 1:model.q
%              gSX = SXGradXpoint(i,j,X);       
%              gSX = gSX*2;
%              S_ufDinvm = K_ufDinvm * gSX;
%              C = K_uuInvK_uf *gSX;
%              D = model.K_uf'* C;
%              gXs(i, j) =  gXs(i, j) - model.mu * sum(sum((model.Ainv*K_ufDinvm).*S_ufDinvm))*model.beta;
               gXs(i, j) =  gXs(i, j) - model.mu * sum(sum(model.Ainv*K_ufDinvm)) *model.beta;
            end
     end
            gX = gXk + gXs;
    end
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

% function Suu = localsimilarity(X_u) 
%     X = X_u;
%     nsq=sum(X.^2,2);
%     K=bsxfun(@minus,nsq,(2*X)*X.');
%     K=bsxfun(@plus,nsq.',K);
%     wi2=  0.5;
%     Suu = exp(-K*wi2);
% end

% function gK = localCovarianceGradients(model)
% % K = kernCompute(model.kern, model.X, model.X);
% gK = model.mu*model.invK_uu*model.SX*model.invK_uu;
% end  

% function gSKuu = localConstraintGradients(model)
% K = model.K_uf'*model.invK_uu*model.K_uf;
% gSKuu = model.mu*invK*model.SX*invK;
% end

% function gSX = GradSX(model)
% X = model.X;
% gSX = cell(size(X));
% N = model.N;
% for i = 1:size(X, 1);
%     for j = 1:size(X, 2);
%       gSX{i,j} = SXGradXpoint(i,j,X,N);
%     end
% end
% end  


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


% function gSX = SXGradXpoint(i,j, X)
% gSX =zeros(size(X,1));
% x = X(:,j);
% gSX1 =zeros(size(X,1));
% gSX2 =zeros(size(X,1));
% gSX1(:, i) = x;
% gSX2(i, :) = x';  
% gSX = gSX1 + gSX2;
% end

% function gKX = kernelGradX(model)
% X = model.X;
% gKX = cell(size(X));
% N = model.N;
% for i = 1:N;
%     for j = 1:size(X, 2);
%       gKX{i,j} = KXGradpoint(model.kern,i,j,X);
%     end
% end
% end  

%rbf kernel
function gKX = KXGradpoint(kern,i,j,X)
gKX = zeros(size(X,1));
x = X(i,:);
n2 = dist2(X, x);
wi2 = (0.5 * kern.comp{1}.inverseWidth);
% wi2 = 0.5;
rbfPart = kern.comp{1}.variance*exp(-n2*wi2);
for k = 1: size(X,1) 
    if (k == i) 
%         gKX(:, i) = kern.comp{1}.inverseWidth*(X(:, j) - x(j)).*rbfPart;
        gKX(:, i) = kern.comp{1}.inverseWidth*(X(:, j) - x(j)).*rbfPart;

         gKX(k, :) = gKX(:, k)';
    end
end
if isfield(kern, 'isNormalised') && (kern.isNormalised == true)
    gKX = gKX * sqrt(kern.inverseWidth/(2*pi));
end
end


%rbfard kernel
%{
function gKX = KXGradpoint(kern,i,j,X)
scales = sparse(sqrt(diag(kern.comp{1}.inputScales)));
gKX = zeros(size(X,1));
x = X(i,:);
n2 = dist2(X*scales, x*scales);
% wi2 = (0.5 * kern.comp{1}.inverseWidth);
wi2 = (0.5.* kern.comp{1}.inverseWidth);
rbfPart = kern.comp{1}.variance*exp(-n2*wi2);
for k = 1: size(X,1) 
    if (k == i) 
%         gKX(:, i) = kern.comp{1}.inverseWidth*(X(:, j) - x(j)).*rbfPart;
        gKX(:, i) = kern.comp{1}.inverseWidth*kern.comp{1}.inputScales(j)*(X(:, j) - x(j)).*rbfPart;

         gKX(k, :) = gKX(:, k)';
    end
end
if isfield(kern, 'isNormalised') && (kern.isNormalised == true)
    gKX = gKX * sqrt(kern.inverseWidth/(2*pi));
end
end
%}