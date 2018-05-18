function g = sgplvmLogLikeGradients_tr(model,verbose)

% SGPLVMLOGLIKEGRADIENTS Compute the gradients for the SGPLVM.
% FORMAT
% DESC Returns the fradients of the log likelihood with respect to
% the parameters of the sgplvm model parameters and the latent
% positions
% ARG model : sgplvm model
% RETURN g : the gradients of the latent positions ( or
% back-constraint parameters) and the parameters of the sgplvm
% model
%
% SEEALSO : sgplvmLogLikelihood, sgplvmCreate,
% modelLogLikeGradients
%
% COPYRIGHT : Neil D. Lawrence, Carl Henrik Ek, 2007, 2009
%
% MODIFICATIONS : Mathieu Salzmann, Carl Henrik Ek, 2009
%
% MODIFICATIONS : Guoli Song, 2018

% SGPLVM

if(nargin<2)
  verbose = false;
end
if(verbose)
  part_name = {'GP','kern','Ind','Const','Lat','P/Dyn','Rank_A','Rank_G','CCA'};
  part_norm = zeros(1,length(part_name));
  part_id = 1;
end


% define chunk id
gX_ind = 3;
gX_u_ind = 2;
gModel_ind = 1;

% calculate gradients for each model
g_full = cell(1,model.numModels);
for(i = 1:1:model.numModels)
  g_full{i} = cell(1,3);
  [g_full{i}{gModel_ind} g_full{i}{gX_u_ind} g_full{i}{gX_ind}] = gpLogLikeGradients(model.comp{i});
end

if(verbose)
  tmp = zeros(1,model.N*model.q);
  for(i = 1:1:model.numModels)
    tmp = tmp + reshape(g_full{i}{gX_ind},1,model.N*model.q);
  end
  part_norm(part_id) = calculateNorm(reshape(tmp,model.N,model.q));
  part_id = part_id + 1;
end

% calculate gradients for kernel F-norm constraints
gkernel = cell(1,model.numModels);
if model.kernelFconstraints
        for(i = 1:1:model.numModels)
          model.comp{i}.lambda = model.lambda;
          gkernel{i} = cell(1,2);
          [gkernel{i}{gModel_ind}, gkernel{i}{gX_u_ind}, gkernel{i}{gX_ind}] = FastkernelFconstraintGradients(model.comp{i});
%           [gkernel{i}{gModel_ind} gkernel{i}{gX_u_ind} gkernel{i}{gX_ind}] = kernelFconstraintGradients(model.comp{i});
          g_full{i}{gX_ind} = bsxfun(@plus, g_full{i}{gX_ind}, gkernel{i}{gX_ind});
          g_full{i}{gX_u_ind} = bsxfun(@plus, g_full{i}{gX_u_ind}, gkernel{i}{gX_u_ind});
          g_full{i}{gModel_ind} = bsxfun(@plus, g_full{i}{gModel_ind}, gkernel{i}{gModel_ind});
        end
end

% calculate gradients for kernel L21 norm constraints
gkernel = cell(1,model.numModels);
if model.kernelSxL21norm  
        for(i = 1:1:model.numModels)
          model.comp{i}.SX = model.SX;
          model.comp{i}.mu = model.mu;
          gkernel{i} = cell(1,2);
%           [gkernel{i}{gModel_ind} gkernel{i}{gX_u_ind} gkernel{i}{gX_ind}] = Fastkernell21constraintGradients(model.comp{i});
%           [gkernel{i}{gModel_ind} gkernel{i}{gX_ind}] = kernell21constraintGradients(model.comp{i});
          [gkernel{i}{gModel_ind}, gkernel{i}{gX_u_ind}] = Fastkernell21Gradients_xu(model.comp{i});
%           g_full{i}{gX_ind} = bsxfun(@plus, g_full{i}{gX_ind}, gkernel{i}{gX_ind});
          g_full{i}{gX_u_ind} = bsxfun(@plus, g_full{i}{gX_u_ind}, gkernel{i}{gX_u_ind});
          g_full{i}{gModel_ind} = bsxfun(@plus, g_full{i}{gModel_ind}, gkernel{i}{gModel_ind});
        end
end

% calculate gradients for kernel trace constraints
gkernel = cell(1,model.numModels);
if model.kernelSxtrace  
        for(i = 1:1:model.numModels)
          model.comp{i}.SX = model.SX;
          model.comp{i}.mu = model.mu;
          gkernel{i} = cell(1,2);
%           [gkernel{i}{gModel_ind}, gkernel{i}{gX_u_ind}, gkernel{i}{gX_ind}] = FastkerneltraceGradients(model.comp{i});
%           [gkernel{i}{gModel_ind}, gkernel{i}{gX_ind}] = kerneltraceGradients(model.comp{i});
%           g_full{i}{gX_ind} = bsxfun(@plus, g_full{i}{gX_ind}, gkernel{i}{gX_ind});
          [gkernel{i}{gModel_ind}, gkernel{i}{gX_u_ind}] = FastkerneltraceGradients_xu(model.comp{i});
          g_full{i}{gX_u_ind} = bsxfun(@plus, g_full{i}{gX_u_ind}, gkernel{i}{gX_u_ind});
          g_full{i}{gModel_ind} = bsxfun(@plus, g_full{i}{gModel_ind}, gkernel{i}{gModel_ind});
        end
end

if(verbose)
  part_norm(part_id) = calculateNorm(reshape(tmp,model.N,model.q));
  part_id = part_id + 1;
end

% prior on inducing points
for(i = 1:1:model.numModels)
  if(isfield(model.comp{i},'inducingPrior'))
    if(~isempty(model.comp{i}.inducingPrior))
      g_full{i}{gX_u_ind} = g_full{i}{gX_u_ind} + priorGradient(model.comp{i}.inducingPrior,model.comp{i}.X_u);
    end
  end
end

if(verbose)
  part_id = part_id + 1;
end

% handle fixInducing points
for(i = 1:1:model.numModels)
  for(j = 1:1:model.q)
    if(model.inducing_id(i,j) && model.comp{i}.fixInducing)
      g_full{i}{gX_ind}(model.comp{i}.inducingIndices,j) = g_full{i}{gX_ind}(model.comp{i}.inducingIndices,j) + g_full{i}{gX_u_ind}(:,j);
    end
  end
end

% constraint part
gX_constraints_dim = zeros(model.N,model.q);
if(isfield(model,'constraints')&&~isempty(model.constraints))
  for(i = 1:1:model.constraints.numConstraints)
    gX_constraints_dim = gX_constraints_dim + constraintLogLikeGradients(model.constraints.comp{i},model.X);
  end
end

if(verbose)
  part_norm(part_id) = calculateNorm(reshape(gX_constraints_dim,model.N,model.q));
  part_id = part_id + 1;
end

% compute gradients for the latent positions
g = [];
gX_dyn_dim = zeros(model.N,model.q);
g_w = cell(1,model.q);
for(i = 1:1:model.q)  
  % prior
  %gX_dyn_dim = zeros(model.N,model.q);
  gX_prior_dim = zeros(model.N,1);
  
  if(model.dynamic)
    ind = find(model.dynamic_id(:,i));
  else
    ind = [];
  end
  if(~isempty(ind))
    for(j = 1:1:length(ind))
      if(isfield(model.dynamics.comp{ind(j)},'balancing'))
	balancing = model.dynamics.comp{ind(j)}.balancing;
      else
	balancing = 1;
      end
      if(isfield(model.dynamics.comp{ind(j)},'indexOut')&&~isempty(model.dynamics.comp{ind(j)}.indexOut))
	gX_dyn_dim(:,model.dynamics.comp{ind(j)}.indexOut) = gX_dyn_dim(:,model.dynamics.comp{ind(j)}.indexOut) +...
	    balancing.*modelLatentGradients(model.dynamics.comp{ind(j)});
      else
	gX_dyn_dim = gX_dyn_dim + balancing.*modelLatentGradients(model.dynamics.comp{ind(j)});
      end
    end
    gX_dim = gX_dyn_dim(:,i);
  else
    ind = find(model.generative_id(:,i))';
    for(j = 1:1:length(ind))
      if(isfield(model.comp{ind(j)},'prior')&&~isempty(model.comp{ind(j)}.prior))
	gX_prior_dim = gX_prior_dim + priorGradient(model.comp{ind(j)}.prior,model.X(:,i));
      end
    end
    if(size(gX_prior_dim,2)==model.q)
      gX_dim = gX_prior_dim(:,i);
    else
      gX_dim = gX_prior_dim(:,1);
    end
  end
  
  if(isfield(model,'constraints')&&~isempty(model.constraints))
    gX_dim = gX_dim + gX_constraints_dim(:,i);
  end
  
  % back-constraints
  ind = find(model.back_id(:,i));
  if(~isempty(ind))
    % model ind back-constrains dimension i
    g_w{i} = modelOutputGrad(model.comp{ind}.back,model.comp{ind}.y,i);
    gX_dim = g_w{i}(:,:)'*gX_dim;

    % add gradients from models generated by dimension i
    ind_gen = find(model.generative_id(:,i));
    for(j = 1:1:length(ind_gen))
      gX_dim = gX_dim + g_w{i}(:,:)'*(g_full{ind_gen(j)}{gX_ind}(:,i));
    end
  else
    % dimension not back-constrained
    ind_gen = find(model.generative_id(:,i));
    for(j = 1:1:length(ind_gen))
      gX_dim = gX_dim + g_full{ind_gen(j)}{gX_ind}(:,i);
    end
  end
  g = [g gX_dim(:)'];
end

if(verbose)
  part_norm(part_id) = calculateNorm(g(:));
  part_id = part_id + 1;

  part_norm(part_id) = calculateNorm(reshape(gX_dyn_dim,model.N,model.q));
  part_id = part_id + 1;
end

% FOLS MODEL

% rank constraints
[g_rank g_rank_alpha g_rank_gamma] = rankLogLikeGradients(model);

if(model.back&&~isempty(g_rank))
  g_rank_back = [];
  g_rank_alpha_back = [];
  g_rank_gamma_back = [];
  for(i = 1:1:model.q)
    ind = find(model.back_id(:,i));
    if(~isempty(ind))
      % dimension i back-constrained apply chain-rule on gradients
      tmp = g_w{i}(:,:)'*g_rank(:,i);
      g_rank_back = [g_rank_back tmp'];
      
      tmp = g_w{i}(:,:)'*g_rank_alpha(:,i);
      g_rank_alpha_back = [g_rank_alpha_back tmp'];
      
      tmp = g_w{i}(:,:)'*g_rank_gamma(:,i);
      g_rank_gamma_back = [g_rank_gamma_back tmp'];
    else
      % use gradients direct
      g_rank_back = [g_rank_back g_rank(:,i)'];
      g_rank_alpha_back = [g_rank_alpha_back g_rank_alpha(:,i)'];
      g_rank_gamma_back = [g_rank_gamma_back g_rank_gamma(:,i)'];
    end
  end
  g_rank = g_rank_back;
  g_rank_alpha = g_rank_alpha_back;
  g_rank_gamma = g_rank_gamma_back;
end

if(~isempty(g_rank))
  g = g + g_rank(:)';
end

if(verbose)
  if(exist('g_rank_alpha','var')&&~isempty(g_rank_alpha))
    part_norm(part_id) = calculateNorm(g_rank_alpha);
  end
  part_id = part_id + 1;
  if(exist('g_rank_gamma','var')&&~isempty(g_rank_gamma))
    part_norm(part_id) = calculateNorm(g_rank_gamma);
  end
  part_id = part_id + 1;
end

% orthogonality constraints
g_ortho = orthoLogLikeGradients(model);

if(model.back&&~isempty(g_ortho))
  g_ortho_back = [];
  for(i = 1:1:model.q)
    ind = find(model.back_id(:,i));
    ind_dim = model.N*(i-1)+1:1:model.N*i;
    if(~isempty(ind))
      % dimension i back-constrained apply chain-rule on gradients
      tmp = g_w{i}(:,:)'*g_ortho(ind_dim)';
      g_ortho_back = [g_ortho_back tmp'];
    else
      % use gradients direct
      g_ortho_back = [g_ortho_back g_ortho(ind_dim)];
    end
  end
  g_ortho = g_ortho_back;
end

if(~isempty(g_ortho))
  g = g + g_ortho;
end

if(verbose)
  if(exist('g_ortho','var')&&~isempty(g_ortho))
    part_norm(part_id) = calculateNorm(reshape(g_ortho,size(g,1),size(g,2)));
  end
  part_id = part_id + 1;
end

% GP's (independent part)
for(i = 1:1:model.numModels)
  switch model.comp{i}.approx
   case 'ftc'
    g = [g g_full{i}{gModel_ind}];
   case {'dtc','fitc','pitc'}
    if(model.comp{i}.fixInducing)
      g = [g g_full{i}{gModel_ind}];
    else
      if(isfield(model.comp{i},'inducingPrior') & ~isempty(model.comp{i}.inducingPrior))
	g_full{i}{gX_u_ind} = g_full{i}{gX_u_ind} + priorGradient(model.inducingPrior, model.comp{i}.X_u);
	g = [g g_full{i}{gX_u_ind}(:)' g_full{i}{gModel_ind}];
      else
	g = [g g_full{i}{gX_u_ind}(:)' g_full{i}{gModel_ind}];
      end
    end
   otherwise
    error('Unkown Approximation');
  end
end

% gradients of parameters of dynamics
if(model.dynamic)
  for(i = 1:1:model.dynamics.numModels)
    g = [g modelLogLikeGradients(model.dynamics.comp{i})];
  end
end


if(verbose)
  fprintf('\nGradient Norms:\n');
  for(i = 1:1:length(part_norm))
    fprintf('%s:\t\t%3.3f\n',part_name{i},100*(part_norm(i)./sum(part_norm)));
  end
  fprintf('\n');
end


return

function g = calculateNorm(g)

g = norm(g(:));

return
