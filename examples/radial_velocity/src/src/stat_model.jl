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
  jitter_sq = RvModelKeplerian.num_jitters >=1 ? extract_jitter(p)^2 : 0.
  chisq = zero(eltype(p))
  log_normalization = -0.5*length(t)*log(2pi)
  for i in 1:length(t)
    model::eltype(p) = calc_model_rv(p,t[i])
    sigma_eff_sq = so[i]^2+jitter_sq
    chisq += abs2(model-o[i])/sigma_eff_sq
    log_normalization -= 0.5*log(sigma_eff_sq)
  end
  return -0.5*(chisq)+log_normalization
end

function plogprior(p::Vector) 
  num_pl = num_planets(p)
  @assert num_pl >= 1
  if !is_valid(p) return -Inf end  # prempt model evaluation
  logp = zero(eltype(p))
  logp -= 2*log(2pi)
  const max_period = 10000.0
  const max_amplitude = 10000.0
  for plid in 1:num_pl
    P::eltype(p) = extract_period(p,plid=plid)
    K::eltype(p) = extract_amplitude(p,plid=plid)
    logp += -log((1+P/P0::Float64)*log1p(max_period/P0::Float64)* 
                 (1+K/K0::Float64)*log1p(max_amplitude/K0::Float64) )

  end
  if RvModelKeplerian.num_jitters >=1
     const max_jitter = 10000.0
     jitter::eltype(p) = extract_jitter(p)
     logp += -log((1+jitter/Jitter0::Float64)*log1p(max_jitter/Jitter0::Float64))
  end
 
  return logp::eltype(p)
end

plogtarget(p::Vector) = ploglikelihood(p) + plogprior(p)




