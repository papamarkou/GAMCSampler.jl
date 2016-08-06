function mahalanobis_distance(x::Vector, y::Vector, M::Matrix)
  z = x-y
  dot(z, M*z)
end

in_mahalanobis_contour(x::Vector, y::Vector, M::Matrix, n::Integer, a::Real=0.95) =
  mahalanobis_distance(x, y, M) < quantile(Chisq(n), a)

mahalanobis_update!(sstate::MuvASMMALAState, pstate::ParameterState{Continuous, Multivariate}, a::Real=0.95) =
  sstate.presentupdatetensor =
    !in_mahalanobis_contour(sstate.pstate.value, sstate.p0value, pstate.tensorlogtarget/sstate.tune.step, pstate.size)
