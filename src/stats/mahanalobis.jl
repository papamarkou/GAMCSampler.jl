function mahalanobis_distance(x::Vector, y::Vector, M::Matrix)
  z = x-y
  dot(z, M*z)
end

in_mahalanobis_contour(x::Vector, y::Vector, M::Matrix, n::Integer, a::Real=0.95) =
  mahalanobis_distance(x, y, M) < quantile(Chisq(n), a)

# function in_mahalanobis_contour(sstate::MuvASMMALAState, pstate::ParameterState{Continuous, Multivariate}, a::Real=0.95)
#   sstate.presentupdatetensor =
#     if dot(
#       sstate.pstate.value-pstate.value,
#       pstate.tensorlogtarget*(sstate.pstate.value-pstate.value)
#     )/sstate.tune.step > quantile(Chisq(pstate.size), a)
#       true
#     else
#       false
#     end
# end
