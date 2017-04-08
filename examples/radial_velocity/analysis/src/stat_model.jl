# Statistical model for analyzing Radial Velocity observations
# Assumes existance of data in arrays times, obs, sigma_obs
# Assumes existance of function calc_model_rv(theta, time)

export ploglikelihood, plogprior, plogtarget
export set_times, set_obs, set_sigma_obs

# Observational data to compare model to
global times # ::Array{Float64,1}
global obs # ::Array{Float64,1}
global sigma_obs #::Array{Float64,1}

# Functions to set global data within module
function set_times(t::Array{Float64,1}) global times = t   end
function set_obs(o::Array{Float64,1})  global obs = o   end
function set_sigma_obs(so::Array{Float64,1}) global sigma_obs = so  end

function ploglikelihood(p::Vector)
  num_pl = num_planets(p)
  @assert num_pl >= 1
  if !is_valid(p) return -Inf end  # prempt model evaluation
  # Set t, o, and so to point to global arrays with observational data, while enforcing types
  t::Array{Float64,1} = times
  o::Array{Float64,1} = obs
  so::Array{Float64,1} = sigma_obs
  @assert length(times) == length(obs) == length(sigma_obs)
  chisq = zero(eltype(p))
  for i in 1:length(t)
    model::eltype(p) = calc_model_rv(p,t[i])
    #chisq += abs2((model-obs[i])/sigma_obs[i])  # More obvious way, but perhaps less efficient
    chisq += abs2((model-o[i])/so[i])
  end
  return -0.5*(chisq)  # WARNING: unnormalized, assumes fixed sigma_obs
end

function plogprior(p::Vector) 
  num_pl = num_planets(p)
  @assert num_pl >= 1
  if !is_valid(p) return -Inf end  # prempt model evaluation
  logp = zero(eltype(p))
  for plid in 1:num_pl
    P::eltype(p) = extract_period(p,plid=plid)
    K::eltype(p) = extract_amplitude(p,plid=plid)
    logp += -log((1+P/P0::Float64)*(1+K/K0::Float64))  # WARNING: unnormalized
  end
  return logp::eltype(p)
end

plogtarget(p::Vector) = ploglikelihood(p) + plogprior(p)




