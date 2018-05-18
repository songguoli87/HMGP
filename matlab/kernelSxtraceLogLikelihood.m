function [model,ll] = kernelSxtraceLogLikelihood(model,X)

% CONSTRAINTLOGLIKELIHOODLDA Constraint loglikelihood for kernel
% FORMAT
% DESC Returns loglikelihood for constraint
% ARG model : fgplvm model
% ARG X : Latent locations
% RETURN ll : Returns loglikelihood
%
% MODIFICATIONS : Guoli Song, 2018

if nargin < 2
  X = model.X_u;
end

Suu = localsimilarity(X); 
model.invKS = model.invK_uu * Suu;  
ll = model.mu*trace(model.invKS);
ll = -ll;
end


function Suu = localsimilarity(X_u) 
    X = X_u;
    nsq=sum(X.^2,2);
    K=bsxfun(@minus,nsq,(2*X)*X.');
    K=bsxfun(@plus,nsq.',K);
    wi2=  0.5;
    Suu = exp(-K*wi2);
end