# HMGP

Matlab code for harmonized GPLVMs (G. Song, et al., Multimodal Gaussian Process Latent Variable Models with Harmonization, ICCV 2017)

The core code of our model is given in the HMGP/matlab/, where each .m file is named according to the type of the harmonization constraint (F, l21, tr).

Dependencies:

(1) GPmat - Neil Lawrence's GP matlab toolbox: https://github.com/SheffieldML/GPmat

(2) Netlab v.3.3: http://www1.aston.ac.uk/ncrg/


We provide the demo code on PASCAL dataset. 
- pascal1K.mat: the feature representations for image and text.
- pascal1K_sim.mat: the intra-modal similarity feature for image and text.
- pascal1K_cat.mat: the class groundtruth.

You can set options.kernelFconstraints to be true for running the model (e.g., main_hmGPLVM) with the F-norm constraint, options.kernelSxL21norm to be true for running the model with the l21-norm constraint, and options.kernelSxtrace  to be true for running the model with the trace constraint.

