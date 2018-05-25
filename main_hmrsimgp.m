% Demonstrate the hm-RSimGP.
clear;
close all;

%%  Load Data
dataType = 'pascal';
dataSetNames = 'pascal';
load('./data/pascal1k_sim.mat');
load('./data/pascal1K_cat');
L = length(cat.tr);
for i=1:L
    for j=1:L
        SXY_tr(i,j) = (cat.tr(i)==cat.tr(j)); % semantic (dis)similarity
    end
end

Ytr{1} = SX_tr;
Ytr{2} = SY_tr;
Yts{1} = SX_te;
Yts{2} = SY_te;
clear SX_tr SY_tr SX_te SY_te;

if(size(Ytr{1},1)>100)
  approx = 'fitc'; %fully independent training conditional
else
  approx = 'ftc'; % no approximation
end

%%  Learn Initialisation through CCA
load('./data/pascal1K.mat');

fprintf('performing svd...\n');

[~,~,v] = svds(I_tr,128);
Y1 = I_tr * v;

[~,~,v] = svds(T_tr,10);
Y2 = T_tr * v;
clear I_tr T_tr I_te T_te;

Ky1 = Y1*Y1';
Ky2 = Y2*Y2';

% pre-process Kernels
Ky1 = kernelCenter(Ky1);
Ky2 = kernelCenter(Ky2);
Ky1 = Ky1./sum(diag(Ky1));
Ky2 = Ky2./sum(diag(Ky2));
Ky1 = (Ky1+Ky1')./2;
Ky2 = (Ky2+Ky2')./2;
   
[A,B] = canoncorr(Ky1,Ky2);
Y1cca = Ky1*A;
Y2cca = Ky2*B;

Xs = (1/2).*(Y1cca+Y2cca);
X_init = Xs ;
X_init = (X_init-repmat(mean(X_init),size(X_init,1),1))./repmat(std(X_init),size(X_init,1),1);
clear Ky1 Ky2 Y1cca Y2cca A B;

%q = 8;
%X_init = X_init(:,1:q);

%% Create model
options_y1 = fgplvmOptions(approx);
options_y1.optimiser = 'scg2';
options_y1.scale2var1 = true;
options_y1.initX = X_init;
options_y1.prior = [];
model{1} = fgplvmCreate(size(options_y1.initX,2),size(Ytr{1},2),Ytr{1},options_y1);
  
options_y2 = fgplvmOptions(approx);
options_y2.optimiser = 'scg2';
options_y2.scale2var1 = true;
options_y2.initX = X_init;
options_y2.prior = [];
model{2} = fgplvmCreate(size(options_y2.initX,2),size(Ytr{2},2),Ytr{2},options_y2);

options = sgplvmOptions;
options.save_intermediate = inf;
options.name = 'hmrsimgp_cca_test_';
options.initX = zeros(2,size(X_init,2));
options.initX(1,:) = true;
options.initX(2,:) = true;
  
options.kernelFconstraints = false; % hmgplvm-Fnorm
options.kernelSxL21norm = false; % hmgplvm-L21 norm
options.kernelSxtrace = true; % hmgplvm-trace

options.mu = 1e-2;
options.gamma1 = 1e0;  

model = sgplvmCreate_tr(model,[],options);

% add semantic constraints 
options_constraint = constraintOptions('Sim');
options_constraint.lambda1 = 1e-1;
options_constraint.lambda2 = 1e-1;
options_constraint.N = model.N;
options_constraint.q = model.q;
options_constraint.dim = 1:model.q;
options_constraint.SXY = SXY_tr;
model = sgplvmAddConstraint(model,options_constraint);
  
%%  Train model
nr_iters = 300;
model = sgplvmOptimise_tr(model,true,nr_iters,false,false);

%%  Test model
obsMod = 1; % one of the involved sub-models (the one for which we have the data)
infMod = setdiff(1:2, obsMod);
numberTestPoints = size(Yts{obsMod},1);
perm = randperm(size(Yts{obsMod},1));
testInd = perm(1:numberTestPoints);

% image query
XY2pred = zeros(length(testInd), size(X_init,2));
for i=1:length(testInd)
    curInd = testInd(i);
    fprintf('# Testing indice number %d ', curInd);
    fprintf('taken from the image test set\n');
    y1_star = Yts{obsMod}(curInd,:);
    index_in = 1;
    index_out = setdiff(1:2, index_in);     
    % Find p(X_* | Y_*) which is approximated by q(X_*)
    x_star = sgplvmPointOut(model,index_in,index_out,y1_star);      
    XY2pred(curInd,:) = x_star;
end  

% text query
XY1pred = zeros(length(testInd), size(X_init,2));
for i=1:length(testInd)
    curInd = testInd(i);
    fprintf('# Testing indice number %d ', curInd);
    fprintf('taken from the text test set\n');
    y2_star = Yts{infMod}(curInd,:);
    index_in = 2;
    index_out = setdiff(1:2, index_in);        
    x_star = sgplvmPointOut(model,index_in,index_out,y2_star);         
    XY1pred(curInd,:) = x_star;
end  
fprintf(' Finish testing.\n')
    
%% map    
map1 = calculateMAP( XY2pred, XY1pred, cat.te );
str = sprintf( 'The MAP of image as query is %f\n', map1);
disp(str);

map2 = calculateMAP( XY1pred, XY2pred, cat.te );
str = sprintf( 'The MAP of text as query is %f\n', map2);
disp(str);  

avgMAP = (map1 + map2) / 2

fprintf(' done.\n')
