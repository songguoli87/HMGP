function model = sgplvmExpandParam_tr(model,params)

% SGPLVMEXPANDPARAM Expand a parameter vector into a sGP-LVM model.
% FORMAT
% DESC takes an SGPLVM structure and a vector of parameters, and
% fills the structure with the given parameters. Also performs any
% necessary precomputation for likelihood and gradient
% computations, so can be computationally intensive to call.
% ARG model : the sgplvm model to update with parameters
% ARG params : parameter vector
% RETURN model : model with updated parameters
%
% SEEALSO : sgplvmCreate, sgplvmExtractParam, modelExpandParam
%
% COPYRIGHT : Neil D. Lawrence, Carl Henrik Ek, 2007, 2009

% MODIFICATIONS : Guoli Song, 2018

% SGPLVM

% Update Back-Constraints
for i = 1:1:model.q
  param_ind = model.parameter_index{1}{i};
    % dimension not back-constrained
    model.X(:,i) = params(param_ind);
end

% Update latent representation in each model
for(i = 1:1:model.numModels)
  model.comp{i}.X = model.X;
end

% Update Generative Part ( this should take care of approx.)
for(i = 1:1:model.numModels)
  param_ind = model.parameter_index{2}{i};
  if(isfield(model.comp{i},'fixInducing'))
    if(model.comp{i}.fixInducing)
      model.comp{i}.X_u = model.X(model.comp{i}.inducingIndices,:);
    end
  end
  model.comp{i} = gpExpandParam(model.comp{i},params(param_ind));
end

% Update Dynamic Part
if(isfield(model,'dynamic'))
  if(model.dynamic)
    for(i = 1:1:model.dynamics.numModels)
      model.dynamics.comp{i} = modelSetLatentValues(model.dynamics.comp{i},model.X);
      param_ind = model.parameter_index{3}{i};
      if(~isempty(param_ind))	
	model.dynamics.comp{i} = modelExpandParam(model.dynamics.comp{i},params(param_ind));
      else
	model.dynamics.comp{i} = modelExpandParam(model.dynamics.comp{i},[]);
      end
    end
  end
end

% Update Constraint Part
if(isfield(model,'constraints')&&~isempty(model.constraints))
  for(i = 1:1:model.constraints.numConstraints)
    model.constraints.comp{i} = constraintExpandParam(model.constraints.comp{i}, model.X);
  end
end

if model.kernelFconstraints    
    X = model.X;
    nsq=sum(X.^2,2);
    K=bsxfun(@minus,nsq,(2*X)*X.');
    K=bsxfun(@plus,nsq.',K);
    wi2=  (.5 ./model.gamma1);
    K=exp(-K*wi2);
    model.SX = K;
    clear K nsq;
end

if model.kernelSxL21norm  
    X = model.X;
    nsq=sum(X.^2,2);
    K=bsxfun(@minus,nsq,(2*X)*X.');
    K=bsxfun(@plus,nsq.',K);
    wi2=  (.5 ./model.gamma1);
    K=exp(-K*wi2);
    model.SX = K;
    clear K nsq;
end

if model.kernelSxtrace  
    X = model.X;
    nsq=sum(X.^2,2);
    K=bsxfun(@minus,nsq,(2*X)*X.');
    K=bsxfun(@plus,nsq.',K);
    wi2=  (.5 ./model.gamma1);
    K=exp(-K*wi2);
    model.SX = K;
    clear K nsq;    
end



