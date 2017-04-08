# Code to extract parameters from unlabeled parameter vector

# model assumes model parameters = [ Period, K, k, h, M0 ] for each planet followed by each RV offset
const num_param_per_planet = 5
pl_offset(plid::Integer) = (plid-1)*num_param_per_planet
num_planets(theta::Vector) = floor(Int64,length(theta)//num_param_per_planet)  # WARNING: Assumes num_rvoffsets<num_param_per_planet
num_obs_offsets(theta::Vector) = length(theta)-num_planets(theta)*num_param_per_planet
obs_offset(theta::Vector,obsid::Integer = 1) = num_planets(theta)*num_param_per_planet+obsid

# constants defining unit system/scale for modified Jeffrys priors
const P0 = 1.0  # units of days
const K0 = 1.0  # units of m/s

set_period(theta::Vector, P; plid::Integer = 1) = theta[1+pl_offset(plid)] = log(1+P/P0)
set_amplitude(theta::Vector, K; plid::Integer = 1) = theta[2+pl_offset(plid)] = log(1+K/K0)
set_ecosw(theta::Vector, h; plid::Integer = 1) = theta[3+pl_offset(plid)] = h
set_esinw(theta::Vector, k; plid::Integer = 1) = theta[4+pl_offset(plid)] = k
set_w_plus_mean_anomaly_at_t0(theta::Vector, wpM0; plid::Integer = 1) = theta[5+pl_offset(plid)] = wpM0
set_rvoffset(theta::Vector, C; obsid::Integer = 1) = theta[obs_offset(theta,obsid)] = C
function set_ew(theta::Vector, e, w; plid::Integer = 1) 
  set_ecosw(theta,e*cos(w),plid=plid)
  set_esinw(theta,e*sin(w),plid=plid) 
end
function set_ewM0(theta::Vector, e, w, M0; plid::Integer = 1) 
  set_ecosw(theta,e*cos(w),plid=plid) 
  set_esinw(theta,e*sin(w),plid=plid) 
  set_w_plus_mean_anomaly_at_t0(theta,w+M0,plid=plid)
end
function set_ewM0(theta::Vector, P, K, e, w, M0; plid::Integer = 1) 
  set_period(theta,P,plid=plid)
  set_amplitude(theta,K,plid=plid) 
  set_ecosw(theta,e*cos(w),plid=plid)
  set_esinw(theta,e*sin(w),plid=plid) 
  set_w_plus_mean_anomaly_at_t0(theta,w+M0,plid=plid)
end


extract_period(theta::Vector; plid::Integer = 1) = P0*(exp(theta[1+pl_offset(plid)])-1)
extract_amplitude(theta::Vector; plid::Integer = 1) = K0*(exp(theta[2+pl_offset(plid)])-1)
extract_ecosw(theta::Vector; plid::Integer = 1) = theta[3+pl_offset(plid)]
extract_esinw(theta::Vector; plid::Integer = 1) = theta[4+pl_offset(plid)]
extract_w(theta::Vector; plid::Integer = 1) = atan(extract_esinw(theta,plid), extract_ecosw(theta,plid))
extract_w_plus_mean_anomaly_at_t0(theta::Vector; plid::Integer = 1) = theta[5+pl_offset(plid)]
extract_mean_anomaly_at_t0(theta::Vector; plid::Integer = 1) = extract_w_plus_mean_anomaly_at_t0(theta,plid)-extract_w(theta,plid)
extract_rvoffset(theta::Vector; obsid::Integer = 1) = theta[obs_offset(theta,obsid)]
function extract_ewM0(theta::Vector; plid::Integer = 1) 
  h = extract_ecosw(theta,plid=plid)
  k = extract_esinw(theta,plid=plid) 
  wpM0 = extract_w_plus_mean_anomaly_at_t0(theta,plid=plid)
  e = sqrt(h*h+k*k)
  w = atan2(k,h)
  M0 = wpM0-w
  return (e,w,M0)
end

function extract_PKhkM(theta::Vector; plid::Integer = 1) 
  local P,K,ecosw,esinw,M0
  P = extract_period(theta, plid=plid)
  K = extract_amplitude(theta, plid=plid)
  ecosw = extract_ecosw(theta, plid=plid)
  esinw = extract_esinw(theta, plid=plid)
  M0 = extract_w_plus_mean_anomaly_at_t0(theta, plid=plid) - atan2(esinw,ecosw)
  return (P,K,ecosw,esinw,M0)
end

function extract_PKewM(theta::Vector; plid::Integer = 1) 
  local P,K,e,w,M0
  P = extract_period(theta, plid=plid)
  K = extract_amplitude(theta, plid=plid)
  (e,w,M0) = extract_ewM0(theta, plid=plid)
  return (P,K,e,w,M0)
end

# check if parameters are valid, i.e., e<1
function is_valid(p::Vector)
  local num_pl,j,k,ecc_sq
  num_pl = num_planets(p)
  @assert length(num_pl) >= 1
  for plid in 1:num_pl
    h = extract_ecosw(p,plid=plid)
    k = extract_esinw(p,plid=plid)
    ecc_sq = h*h+k*k
    if !(ecc_sq<1.0) 
      return false
    end
  end
  return true
end

export num_planets, num_obs_offsets
export set_period, set_amplitude, set_ecosw, set_esinw, set_w_plus_mean_anomaly_at_t0, set_mean_anomaly_at_t0, set_rvoffset, set_ew, set_ewM0
export extract_period, extract_amplitude, extract_ecosw, extract_esinw, extract_w_plus_mean_anomaly_at_t0, extract_mean_anomaly_at_t0, extract_ewM0, extract_PKhkM, extract_PKewM, extract_rvoffset
export extract_PKhkM, isvalid



