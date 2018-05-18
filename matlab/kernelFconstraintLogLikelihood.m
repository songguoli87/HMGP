function ll = kernelFconstraintLogLikelihood(model,X)

% CONSTRAINTLOGLIKELIHOODLDA Constraint loglikelihood for kernel
% FORMAT
% DESC Returns loglikelihood for constraint
% ARG model : fgplvm model
% ARG X : Latent locations
% RETURN ll : Returns loglikelihood
%
% MODIFICATIONS : Guoli Song, 2018

if nargin < 2
  X = model.X;
end
KX = kernCompute(model.kern, X, X);
H = KX - model.SX;
ll = 0.5*norm(H,'fro');% F ·¶Êý
ll = model.mu*ll;
ll = -ll;
return;