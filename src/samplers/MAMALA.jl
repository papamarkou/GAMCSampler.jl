### MAMALA state subtypes

abstract MAMALAState <: LMCSamplerState

## MuvMAMALAState holds the internal state ("local variables") of the MAMALA sampler for multivariate parameters

type MuvMAMALAState <: MAMALAState
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
  presentupdatetensor::Bool
  pastupdatetensor::Bool
  count::Integer
  updatetensorcount::Integer

  function MuvMAMALAState(
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
    presentupdatetensor::Bool,
    pastupdatetensor::Bool,
    count::Integer,
    updatetensorcount::Integer
  )
    if !isnan(ratio)
      @assert 0 < ratio < 1 "Acceptance ratio should be between 0 and 1"
      @assert sqrttunestep > 0 "Square root of tuned drift step is not positive"
    end
    new(
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
      presentupdatetensor,
      pastupdatetensor,
      count,
      updatetensorcount
    )
  end
end

MuvMAMALAState(pstate::ParameterState{Continuous, Multivariate}, tune::MCTunerState=MAMALAMCTune()) =
  MuvMAMALAState(
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
  false,
  false,
  0,
  0
)

### Manifold adaptive Metropolis-adjusted Langevin algorithm (MAMALA)

immutable MAMALA <: LMCSampler
  driftstep::Real
  update!::Function
  transform::Union{Function, Void}
  t0::Integer

  function MAMALA(driftstep::Real, update!::Function, transform::Union{Function, Void}, t0::Integer)
    @assert driftstep > 0 "Drift step is not positive"
    @assert t0 > 0 "t0 is not positive"
    new(driftstep, update!, transform, t0)
  end
end

MAMALA(driftstep::Real=1.; update::Function=rand_update!, transform::Union{Function, Void}=nothing, t0::Integer=3) =
  MAMALA(driftstep, update, transform, t0)

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

tuner_state(sampler::MAMALA, tuner::MAMALAMCTuner) =
  MAMALAMCTune(
    BasicMCTune(NaN, 0, 0, tuner.smmalatuner.period),
    BasicMCTune(NaN, 0, 0, tuner.malatuner.period),
    BasicMCTune(sampler.driftstep, 0, 0, tuner.totaltuner.period)
  )

function sampler_state(
  parameter::Parameter{Continuous, Multivariate},
  sampler::MAMALA,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector
)
  sstate = MuvMAMALAState(generate_empty(pstate, parameter.diffmethods, parameter.diffopts), tuner_state(sampler, tuner))
  sstate.sqrttunestep = sqrt(sampler.driftstep)
  sstate.lastmean = copy(pstate.value)
  sstate.oldinvtensor = inv(pstate.tensorlogtarget)
  sstate.cholinvtensor = ctranspose(chol(Hermitian(sstate.oldinvtensor)))
  sstate.oldfirstterm = sstate.oldinvtensor*pstate.gradlogtarget
  sstate.presentupdatetensor = true
  sstate.pastupdatetensor = false
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
  sstate.oldinvtensor = inv(pstate.tensorlogtarget)
  sstate.cholinvtensor = chol(sstate.oldinvtensor, Val{:L})
  sstate.oldfirstterm = sstate.oldinvtensor*pstate.gradlogtarget
  sstate.presentupdatetensor = true
  sstate.pastupdatetensor = false
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

Base.show(io::IO, sampler::MAMALA) = print(io, "MAMALA sampler: drift step = $(sampler.driftstep)")

Base.show(io::IO, ::MIME"text/plain", sampler::MAMALA) = show(io, sampler)
