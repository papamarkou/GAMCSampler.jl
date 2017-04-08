

# Compute Eccentric Anomally from Mean Anomally
# WARNING: Only valid for small e.  Not convergent for e>~0.66
# Therefore, this should only be used for testing purposes, not general science applications
function ecc_anom_bessel_series_approx{T}(mean_anom::T, ecc::T; tol::Real = 1.e-12)
  sinM = sin(mean_anom)
  sin2M = sin(2*mean_anom)
  sin3M = sin(3*mean_anom)
  sin4M = sin(4*mean_anom)
  E = mean_anom+ecc*(sinM+ecc*(0.5*sin2M+ecc*((3*sin3M-sinM)/8+ecc*((sin4M-0.5*sin2M)/3))))
end

#=  Compute Eccentric Anomally from Mean Anomally robustly
    Initial guess for eccentric anomaly for use by itterative solvers of Kepler's equation
    Based on "The Solution of Kepler's Equations - Part Three"
            Danby, J. M. A. (1987) Journal: Celestial Mechanics, Volume 40, Issue 3-4, pp. 303-312
            1987CeMec..40..303D
=# 
function ecc_anom_init_guess_danby{T}(M::T, ecc::T)
    local k, E
    const k = convert(T,0.85)
    if(M<zero(T)) M += 2pi end
    E = (M<pi) ? M + k*ecc : M - k*ecc;
end

#= Update the current guess for solution to Kepler's equation
   Based on "An Improved Algorithm due to Laguerre for the Solution of Kepler's Equation"
   Conway, B. A.  (1986) Celestial Mechanics, Volume 39, Issue 2, pp.199-211
  1986CeMec..39..199C
=#
function update_ecc_anom_laguerre{T}(E::T, M::T, ecc::T)
  es = ecc*sin(E)
  ec = ecc*cos(E)
  F = (E-es)-M
  Fp = one(T)-ec
  Fpp = es
  const n = 5
  root = sqrt(abs((n-1)*((n-1)*Fp*Fp-n*F*Fpp)))
  denom = Fp>zero(T) ? Fp+root : Fp-root
  return E-n*F/denom
end

# Loop to update the current estimate of the solution to Kepler's equation
function calc_ecc_anom_itterative_laguerre{T}(mean_anom::T, ecc::T; tol::Real = 1.e-8)
    local M, E, E_old
    #M = atan2(sin(mean_anom),cos(mean_anom))  # WARNING: Kludge since missing autodif for mod2pi  # [-pi,pi]
    M = mod2pi(mean_anom)
    E = ecc_anom_init_guess_danby(M,ecc)
    const max_its_laguerre = 200
    for i in 1:max_its_laguerre
       E_old = E
       E = update_ecc_anom_laguerre(E_old, M, ecc)
       if abs(E-E_old)<tol break end
    end
    @assert abs(E-E_old)<tol
    return E
end

# Set the default Kepler solver
calc_ecc_anom = calc_ecc_anom_itterative_laguerre
#=
 function calc_ecc_anom{T}(mean_anom::T, ecc::T; tol::Real = 1.e-8)
  return calc_ecc_anom_itterative_laguerre(mean_anom, ecc, tol=tol)
end
=#

# export calc_ecc_anom

