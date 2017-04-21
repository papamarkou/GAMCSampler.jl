#=
   Provides calc_model_rv(theta, time) 
   Computes the velocity of the star due to the perturbations of multiple planets, as the linear superposition of the Keplerian orbit induced by each planet, i.e., neglecting mutual planet-planet interactions
=#

include("kepler_eqn.jl")       # Code to solve Kepler's equation provides calc_ecc_anom(mean_anom, ecc; tol = 1.e-8)

# Calculate Keplerian velocity of star due to one planet (with parameters displaced by offset)
function calc_rv_pal_one_planet{T}( theta::Array{T,1}, time::Float64; plid::Integer = 1, tol::Real = 1.e-8 )
  #= P = extract_period(theta,offset=offset)
  K = extract_amplitude(theta,offset=offset)
  h = extract_ecosw(theta,offset=offset)   # TODO: check h & k def)
  k = extract_esinw(theta,offset=offset)
  M0 = extract_mean_anomaly_at_t0(theta,offset=offset) =#
  (P,K,h,k,M0) = extract_PKhkM(theta,plid=plid)
  ecc = sqrt(h*h+k*k)
  w = atan2(k,h)
  n = 2pi/P
  #M = mod2pi(time*n-M0)
  M = time*n-M0
  lambda = w+M
  #E = ecc_anom_bessel_series_approx(M,ecc)  # WARNING: This would calling approximate version
  #E = ecc_anom_itterative_laguerre(M,ecc,tol=tol) # WARNING: hardwired particular algorithm
  E = calc_ecc_anom(M,ecc,tol=tol)
  c = cos(lambda+ecc*sin(E))
  s = sin(lambda+ecc*sin(E))
  if ecc >= 1.0
    println("# ERROR in calc_rv_pal_one_planet: ecc>=1.0:  ",theta)
  end
  @assert(0.0<=ecc<1.0)
  j = sqrt((1.0-ecc)*(1.0+ecc))
  #p, q = (ecc == 0.0) ? fill((0.0, 0.0), length(time)) : (ecc.*sin(E), ecc.*cos(E))
  p, q = (ecc == 0.0) ? (zero(T), zero(T)) : (ecc*sin(E), ecc*cos(E))
  a = K/(n/sqrt((1.0-ecc)*(1.0+ecc)))
  zdot = a*n/(1.0-q)*( cos(lambda+p)-k*q/(1.0+j) )
end

# Calculate Keplerian velocity of star due to num_pl planets
function calc_rv_pal_multi_planet{T}( theta::Array{T,1}, time::Float64; tol::Real = 1.e-8)
  zdot = zero(T)
  for plid in 1:num_planets(theta)
      zdot += calc_rv_pal_one_planet(theta,time,plid=plid,tol=tol)
  end
  return zdot
end

# Assumes model parameters = [ Period_1, K_1, k_1, h_1, M0_1,   Period_2, K_2, k_2, h_2, M0_2, ...  C ] 
function calc_model_rv{T}( theta::Array{T,1}, t::Float64; obsid::Integer = 1, tol::Real = 1.e-8)
  Offset = extract_rvoffset(theta, obsid=obsid)
  calc_rv_pal_multi_planet(theta, t, tol=tol) + Offset
end

export isvalid, calc_model_rv


