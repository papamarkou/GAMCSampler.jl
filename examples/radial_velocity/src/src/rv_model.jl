#=
   Likelihhod is chi^2 comparing observations of the radial velocity of the star to a model, 
   where the model computes the velocity of the star as the linear superposition of the Keplerian orbit 
   induced by each planet, i.e., neglecting mutual planet-planet interactions 
   Priors based on SAMSI reference priors from Ford & Gregory 2006 
=#

module RvModelKeplerian

include("rv_model_param.jl")       # Code to convert parameter vector to named parameters
include("utils_internal.jl")       # Manually add code to support autodiff of mod2pi
include("rv_model_keplerian.jl")   # Code to compute RV model for Keplerian orbits
include("stat_model.jl")           # Code to compute target density, plogtarget


end # module RvModelKeplerian

