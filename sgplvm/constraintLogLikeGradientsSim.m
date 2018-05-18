function gX = constraintLogLikeGradientsSim(model,X)

% CONSTRAINTLOGLIKEGRADIENTSSIM Returns gradients of loglikelihood
% for semantic constraints
% FORMAT
% DESC Returns loglikelihood for semantic constraint
% ARG model : fgplvm model
% RETURN options : Returns loglikelihood
%
% SEEALSO : constraintLogLikelihood
%
% COPYRIGHT : Guoli Song, 2018

% rsimgp

gX1 = zeros(model.N, model.q);
gX2 = zeros(model.N, model.q);
gXns1 = zeros(model.N, model.q);
X1 = X;
X2 = X;
gKX= simGradX(X1, X2);

% keep non-simliarity
for i = 1:model.N
  for j = 1:model.q
        gX1(i, j) = (-gKX(:, j, i))'*(1 - model.SXY(:, i));
        if (model.distX (i, j) >= 1)
            gXns1(i, j) = (-gKX(:, j, i))'*(1 - model.SXY(:, i));
        end
        gX1(i, j) = gX1(i, j) - gXns1(i, j);
  end
end     
gX1 = model.lambda1 * gX1;

% keep similarity 
for i = 1:model.N
  for j = 1:model.q
    gX2(i, j) = gKX(:, j, i)'*model.SXY(:, i);
  end
end     
gX2 = model.lambda2 * gX2;

gX = gX1 + gX2;
gX = -gX;
return

% rbfKernGradX
function gKX = simGradX( X, X2)
    gKX = zeros(size(X2, 1), size(X2, 2), size(X, 1));
    for i = 1:size(X, 1);
      gKX(:, :, i) = rbfKernGradXpoint( X(i, :), X2);
    end


function gKX = rbfKernGradXpoint( x, X2)
% RBFKERNGRADXPOINT Gradient with respect to one point of x.
  gKX = zeros(size(X2));
  for i = 1:size(x, 2)
   gKX(:, i) = 4*(x(i) - X2(:, i));
  end


