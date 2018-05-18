function model = constraintCreateSim(options)

% CONSTRAINTCREATESIM Creates a semantic constraint model from a options struct
% FORMAT
% DESC Creates a semantic constraint model from a options struct
% ARG options : options structure as returned by constraintOptions
% RETURN model : the model created
%
% SEEALSO : constraintOptions
%
% MODIFICATIONS : Guoli Song, 2018

% rsimgp

model = {};
model.type = options.type;

if(isfield(options,'lambda1')&&~isempty(options.lambda1))
  model.lambda1 = options.lambda1;
else
  model.lambda1 = 1.0;
end

if(isfield(options,'lambda2')&&~isempty(options.lambda2))
  model.lambda2 = options.lambda2;
else
  model.lambda2 = 1.0;
end

if(isfield(options,'sigma')&&~isempty(options.sigma))
  model.sigma = options.sigma;
else
  model.sigma = 1;
end

if(isfield(options,'SXY')&&~isempty(options.SXY))
  model.SXY = options.SXY;
else
  model.SXY = 1;
end

model.N = options.N;
model.q = options.q;

return