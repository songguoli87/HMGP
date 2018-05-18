function model = constraintExpandParamSim(model,X)

% CONSTRAINTEXPANDPARAMLDA Expands a semantic constraint model
% FORMAT
% DESC Returns expanded model
% ARG model : constraint model
% ARG X : Latent locations
% RETURN model : Returns expanded model
%
% SEEALSO : constraintExpandParam
% MODIFICATIONS : Guoli Song, 2018
% rsimgp

X1 = X;
X2 = X;

SXY = model.SXY;
nPosData = sum(SXY(:));
DXY = 1 - SXY;
nNegData = sum(DXY(:));

distX = dist2(X1, X2);

SimX = distX.* SXY;
DifX = (1 - distX).*DXY;

model.distX = distX;
model.SimX = SimX;
model.DifX = DifX;
model.nPosData = nPosData;
model.nNegData = nNegData;
return