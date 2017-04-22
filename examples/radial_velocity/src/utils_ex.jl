# Function to create param vector using data in P_true, K_true, e_true, w_true, M0_true, and C_true and RvModelKeplerian.num_param_per_planet

function make_param_true(P_true::Vector, K_true::Vector, e_true::Vector, w_true::Vector, M0_true::Vector, C_true::Vector, jitter::Real)
  num_pl = length(P_true)
  @assert num_pl == length(P_true) == length(K_true) == length(e_true) == length(w_true) == length(M0_true)
  nppp = RvModelKeplerian.num_param_per_planet
  @assert 1 <= length(C_true) < nppp
  param = zeros(nppp*num_pl+length(C_true)+RvModelKeplerian.num_jitters)
  for i in 1:num_pl
      set_period(param,P_true[i],plid=i) 
      set_amplitude(param,K_true[i],plid=i) 
      set_ewM0(param,e_true[i],w_true[i],M0_true[i],plid=i) 
  end
  for i in 1:length(C_true)
      set_rvoffset(param,C_true[i],obsid=i)
  end
  if RvModelKeplerian.num_jitters >= 1
     set_jitter(param,jitter)
  end
  return param
end

function make_param_perturb_scale(param::Vector)
  param_perturb_scale = ones(length(param))
  for i in 1:RvModelKeplerian.num_planets(param)
     #set_period(param_perturb_scale,0.0004/extract_period(param,plid=i),plid=i) 
     #set_amplitude(param_perturb_scale,0.002/extract_amplitude(param,plid=i),plid=i) 
     #set_ecosw(param_perturb_scale,0.002,plid=i) 
     #set_esinw(param_perturb_scale,0.002,plid=i) 
     #set_w_plus_mean_anomaly_at_t0(param_perturb_scale,0.004,plid=i)
     set_period(param_perturb_scale,0.0001,plid=i) 
     set_amplitude(param_perturb_scale,0.01,plid=i) 
     set_ecosw(param_perturb_scale,0.05,plid=i) 
     set_esinw(param_perturb_scale,0.05,plid=i) 
     set_w_plus_mean_anomaly_at_t0(param_perturb_scale,pi*0.01,plid=i)
  end
  for i in 1:num_obs_offsets(param)
     #set_rvoffset(param_perturb_scale,0.003,obsid=i)
     set_rvoffset(param_perturb_scale,0.3,obsid=i)
  end
  if RvModelKeplerian.num_jitters >= 1
    #set_jitter(param_perturb_scale,0.03/extract_jitter(param))
    set_jitter(param_perturb_scale,0.01)
  end
  return param_perturb_scale
end

function make_param_true_ex1()
  P_true = 50.0    # Orbital period
  K_true = 20.0    # Velocityi amplitude
  e_true = 0.2     # Oribtal eccentricity
  w_true = pi/4    # Argument of pericenter (i.e., orientation of elipse)
  M0_true = pi/4   # Mean Anomaly at time zero (i.e., where planet is on elipse at t=0)
  C_true = 1.0     # Arbitrary zero point for observatory
  jitter_true = 1. # Jitter
  param_true = make_param_true([P_true],[K_true],[e_true],[w_true],[M0_true],[C_true], jitter_true)
end

function make_param_true_ex2()
  P_true = [40.0, 80.8]     # Orbital period
  K_true = [30.0, 30.0]     # Velocityi amplitude
  e_true = [0.2,   0.2]     # Oribtal eccentricity
  w_true = pi/4*ones(2)     # Argument of pericenter (i.e., orientation of elipse)
  M0_true = pi/4*ones(2)    # Mean Anomaly at time zero (i.e., where planet is on elipse at t=0)
  C_true = [0.0 ]           # Arbitrary zero point for observatory
  jitter_true = 1. # Jitter
  param_true = make_param_true(P_true,K_true,e_true,w_true,M0_true,C_true,jitter_true)
end

