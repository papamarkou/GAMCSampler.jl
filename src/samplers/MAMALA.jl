### MAMALA state subtypes

abstract MAMALAState <: LMCSamplerState

## MuvMAMALAState holds the internal state ("local variables") of the MAMALA sampler for multivariate parameters

type MuvMAMALAState <: MAMALAState
  proposal::Union{MultivariateGMM, AbstractMvNormal, Void}
  pstate::ParameterState{Continuous, Multivariate}
  tune::MCTunerState
  sqrttunestep::Real
  ratio::Real
  μ::RealVector
  lastmean::RealVector
  secondlastmean::RealVector
  newinvtensor::RealMatrix
  oldinvtensor::RealMatrix
  cholinvtensor::RealLowerTriangular
  newfirstterm::RealVector
  oldfirstterm::RealVector
  sqrtminorscale::Real
  w::RealVector
  presentupdatetensor::Bool
  pastupdatetensor::Bool
  count::Integer
  updatetensorcount::Integer

  function MuvMAMALAState(
    proposal::Union{MultivariateGMM, AbstractMvNormal, Void},
    pstate::ParameterState{Continuous, Multivariate},
    tune::MCTunerState,
    sqrttunestep::Real,
    ratio::Real,
    μ::RealVector,
    lastmean::RealVector,
    secondlastmean::RealVector,
    newinvtensor::RealMatrix,
    oldinvtensor::RealMatrix,
    cholinvtensor::RealLowerTriangular,
    newfirstterm::RealVector,
    oldfirstterm::RealVector,
    sqrtminorscale::Real,
    w::RealVector,
    presentupdatetensor::Bool,
    pastupdatetensor::Bool,
    count::Integer,
    updatetensorcount::Integer
  )
    if !isnan(ratio)
      @assert 0 < ratio < 1 "Acceptance ratio should be between 0 and 1"
      @assert sqrttunestep > 0 "Square root of tuned drift step is not positive"
    end

    if !isnan(sqrtminorscale)
      @assert sqrtminorscale >= 0 "Scaling of stabilizing covariance should be non-negative"
    end

    if !isnan(w[1])
      @assert w[1] > 0 "Weight of core mixture component must be positive"
    end
    if !isnan(w[2])
      @assert w[2] >= 0 "Weight of minor mixture component must be non-negative"
    end

    @assert count >= 0 "Number of iterations (count) should be non-negative"

    new(
      proposal,
      pstate,
      tune,
      sqrttunestep,
      ratio,
      μ,
      lastmean,
      secondlastmean,
      newinvtensor,
      oldinvtensor,
      cholinvtensor,
      newfirstterm,
      oldfirstterm,
      sqrtminorscale,
      w,
      presentupdatetensor,
      pastupdatetensor,
      count,
      updatetensorcount
    )
  end
end

MuvMAMALAState(
  pstate::ParameterState{Continuous, Multivariate},
  sqrtminorscale::Real,
  w::RealVector,
  tune::MCTunerState=MAMALAMCTune(),
  lastmean::RealVector=Array(eltype(pstate), pstate.size)
) =
  MuvMAMALAState(
  nothing,
  pstate,
  tune,
  NaN,
  NaN,
  Array(eltype(pstate), pstate.size),
  Array(eltype(pstate), pstate.size),
  Array(eltype(pstate), pstate.size),
  Array(eltype(pstate), pstate.size, pstate.size),
  Array(eltype(pstate), pstate.size, pstate.size),
  RealLowerTriangular(Array(eltype(pstate), pstate.size, pstate.size)),
  Array(eltype(pstate), pstate.size),
  Array(eltype(pstate), pstate.size),
  sqrtminorscale,
  w,
  true,
  false,
  0,
  0
)

### Manifold adaptive Metropolis-adjusted Langevin algorithm (MAMALA)

immutable MAMALA <: LMCSampler
  update!::Function
  transform::Union{Function, Void}
  driftstep::Real
  minorscale::Real # Scaling factor of covariance matrix of the stabilizing mixture component
  c::Real # Non-negative constant with relative small value that determines the mixture weight of the stabilizing component
  t0::Integer

  function MAMALA(
    update!::Function,
    transform::Union{Function, Void},
    driftstep::Real,
    minorscale::Real,
    c::Real,
    t0::Integer
  )
    @assert driftstep > 0 "Drift step is not positive"
    @assert 0 < minorscale "Constant minorscale must be positive, got $minorscale"
    @assert 0 <= c "Constant c must be non-negative"
    @assert t0 > 0 "t0 is not positive"
    new(update!, transform, driftstep, minorscale, c, t0)
  end
end

MAMALA(;
  update::Function=rand_update!,
  transform::Union{Function, Void}=nothing,
  driftstep::Real=1.,
  minorscale::Real=1.,
  c::Real=0.05,
  t0::Integer=3
) =
  MAMALA(update, transform, driftstep, minorscale, c, t0)

### Initialize MAMALA sampler

## Initialize parameter state

function initialize!(
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::MAMALA,
  outopts::Dict
)
  parameter.uptotensorlogtarget!(pstate)
  if sampler.transform != nothing
    pstate.tensorlogtarget = sampler.transform(pstate.tensorlogtarget)
  end
  pstate.diagnosticvalues[1] = false
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"
  @assert all(isfinite(pstate.gradlogtarget)) "Gradient of log-target not finite: initial values out of support"
  @assert all(isfinite(pstate.tensorlogtarget)) "Tensor of log-target not finite: initial values out of support"

  if !isempty(outopts[:diagnostics])
    pstate.diagnostickeys = copy(outopts[:diagnostics])
    pstate.diagnosticvalues = Array(Any, length(pstate.diagnostickeys))
  end
end

## Initialize MAMALA state

set_gmm(
  sampler::MAMALA,
  pstate::ParameterState{Continuous, Multivariate},
  C::RealMatrix
  corescale::Real,
  sqrtminorscale::Real,
  w::RealVector
) =
  MixtureModel([MvNormal(pstate.value, corescale*C), MvNormal(pstate.value, sqrtminorscale)], w)

set_gmm!(
  sstate::MuvAMState,
  sampler::MAMALA,
  pstate::ParameterState{Continuous, Multivariate}
) =
  sstate.proposal = set_gmm(sampler, pstate, sstate.C, sstate.tune.totaltune.step, sstate.sqrtminorscale, sstate.w)

tuner_state(sampler::MAMALA, tuner::MAMALAMCTuner) =
  MAMALAMCTune(
    BasicMCTune(NaN, 0, 0, tuner.smmalatuner.period),
    BasicMCTune(NaN, 0, 0, tuner.amtuner.period),
    BasicMCTune(sampler.driftstep, 0, 0, tuner.totaltuner.period)
  )

function sampler_state(
  parameter::Parameter{Continuous, Multivariate},
  sampler::MAMALA,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector
)
  oldinvtensor = inv(pstate.tensorlogtarget)
  w = [1-sampler.c, sampler.c]

  sstate = MuvMAMALAState(
    generate_empty(pstate, parameter.diffmethods, parameter.diffopts),
    sqrt(sampler.minorscale),
    w,
    tuner_state(sampler, tuner),
    copy(pstate.value)
  )

  sstate.sqrttunestep = sqrt(sampler.driftstep)
  sstate.oldinvtensor[:, :] = inv(pstate.tensorlogtarget)
  sstate.cholinvtensor[:, :] = ctranspose(chol(Hermitian(sstate.oldinvtensor)))
  sstate.oldfirstterm[:] = sstate.oldinvtensor*pstate.gradlogtarget

  sstate
end

### Reset MAMALA sampler

## Reset parameter state

function reset!(
  pstate::ParameterState{Continuous, Multivariate},
  x::RealVector,
  parameter::Parameter{Continuous, Multivariate},
  sampler::MAMALA
)
  pstate.value = copy(x)
  parameter.uptotensorlogtarget!(pstate)
  pstate.diagnosticvalues[1] = false
end

## Reset sampler state

function reset!(
  sstate::MuvMAMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::MAMALA,
  tuner::MCTuner
)
  reset!(sstate.tune, sampler, tuner)
  sstate.sqrttunestep = sqrt(sampler.driftstep)
  sstate.lastmean = copy(pstate.value)
  sstate.presentupdatetensor = true
  sstate.pastupdatetensor = false
  sstate.count = 0
end

### MAMALA schedules

am_only_update!(sstate::MuvMAMALAState, pstate::ParameterState{Continuous, Multivariate}, i::Integer, tot::Integer) =
  sstate.presentupdatetensor = false

cos_update!(
  sstate::MuvMAMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  a::Real=10.,
  b::Real=0.2,
  c::Real=0.2
) =
  sstate.presentupdatetensor = rand(Bernoulli(b*cos(a*i*pi/tot)+c))

mod_update!(
  sstate::MuvMAMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  n::Integer=10
) =
  sstate.presentupdatetensor = mod(i, n) == 0 ? true : false

rand_exp_decay_update!(
  sstate::MuvMAMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  a::Real=10.,
  b::Real=0.
) =
  sstate.presentupdatetensor = rand(Bernoulli(exp_decay(i, tot, a, b)))

rand_lin_decay_update!(
  sstate::MuvMAMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  a::Real=1e-2,
  b::Real=0.
) =
  sstate.presentupdatetensor = rand(Bernoulli(lin_decay(i, tot, a, b)))

rand_pow_decay_update!(
  sstate::MuvMAMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  a::Real=1e-3,
  b::Real=0.
) =
  sstate.presentupdatetensor = rand(Bernoulli(pow_decay(i, tot, a, b)))

rand_quad_decay_update!(
  sstate::MuvMAMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  a::Real=1e-5,
  b::Real=0.
) =
  sstate.presentupdatetensor = rand(Bernoulli(quad_decay(i, tot, a, b)))

rand_update!(
  sstate::MuvMAMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  p::Real=0.5
) =
  sstate.presentupdatetensor = rand(Bernoulli(p))

smmala_only_update!(sstate::MuvMAMALAState, pstate::ParameterState{Continuous, Multivariate}, i::Integer, tot::Integer) =
  sstate.presentupdatetensor = true

### MAMALA show methods

Base.show(io::IO, sampler::MAMALA) = print(
  io,
  "MAMALA sampler: drift step = ",
  sampler.driftstep,
  ", scaling of stabilizing covariance = ",
  sampler.minorscale,
  ", weight of stabilizing mixture component = ",
  sampler.c
)

Base.show(io::IO, ::MIME"text/plain", sampler::MAMALA) = show(io, sampler)
