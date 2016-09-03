### GUM state subtypes

## MuvGUMState holds the internal state ("local variables") of the GUM sampler for multivariate parameters

type MuvGUMState <: MuvPSMMALAState
  pstate::ParameterState{Continuous, Multivariate}
  tune::MCTunerState
  sqrttunestep::Real
  ratio::Real
  μ::RealVector
  newinvtensor::RealMatrix
  oldinvtensor::RealMatrix
  cholinvtensor::RealLowerTriangular
  newfirstterm::RealVector
  oldfirstterm::RealVector
  SST::RealMatrix
  randnsample::RealVector
  η::Real
  presentupdatetensor::Bool
  pastupdatetensor::Bool
  count::Integer
  updatetensorcount::Integer

  function MuvGUMState(
    pstate::ParameterState{Continuous, Multivariate},
    tune::MCTunerState,
    sqrttunestep::Real,
    ratio::Real,
    μ::RealVector,
    newinvtensor::RealMatrix,
    oldinvtensor::RealMatrix,
    cholinvtensor::RealLowerTriangular,
    newfirstterm::RealVector,
    oldfirstterm::RealVector,
    SST::RealMatrix,
    randnsample::RealVector,
    η::Real,
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
      newinvtensor,
      oldinvtensor,
      cholinvtensor,
      newfirstterm,
      oldfirstterm,
      SST,
      randnsample,
      η,
      presentupdatetensor,
      pastupdatetensor,
      count,
      updatetensorcount
    )
  end
end

MuvGUMState(pstate::ParameterState{Continuous, Multivariate}, tune::MCTunerState=PSMMALAMCTune()) =
  MuvGUMState(
  pstate,
  tune,
  NaN,
  NaN,
  Array(eltype(pstate), pstate.size),
  Array(eltype(pstate), pstate.size, pstate.size),
  Array(eltype(pstate), pstate.size, pstate.size),
  RealLowerTriangular(Array(eltype(pstate), pstate.size, pstate.size)),
  Array(eltype(pstate), pstate.size),
  Array(eltype(pstate), pstate.size),
  Array(eltype(pstate), pstate.size, pstate.size),
  Array(eltype(pstate), pstate.size),
  NaN,
  false,
  false,
  0,
  0
)

### Metropolis-adjusted Langevin Algorithm (GUM)

immutable GUM <: PSMMALA
  driftstep::Real
  update!::Function
  transform::Union{Function, Void}
  t0::Integer
  targetrate::Real
  γ::Real

  function GUM(driftstep::Real, update!::Function, transform::Union{Function, Void}, t0::Integer, targetrate::Real, γ::Real)
    @assert driftstep > 0 "Drift step is not positive"
    @assert t0 > 0 "t0 is not positive"
    @assert 0 < targetrate < 1 "Target acceptance rate should be between 0 and 1"
    @assert 0.5 < γ <= 1 "Exponent of stepsize must be greater than 0.5 and less or equal to 1"
    new(driftstep, update!, transform, t0, targetrate, γ)
  end
end

GUM(
  driftstep::Real=1.;
  update::Function=rand_update!,
  transform::Union{Function, Void}=nothing,
  t0::Integer=3,
  targetrate::Real=0.234,
  γ::Real=0.7
) =
  GUM(driftstep, update, transform, t0, targetrate, γ)

### Initialize GUM sampler

## Initialize parameter state

function initialize!(
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::GUM
)
  parameter.uptotensorlogtarget!(pstate)
  if sampler.transform != nothing
    pstate.tensorlogtarget = sampler.transform(pstate.tensorlogtarget)
  end
  pstate.diagnosticvalues[1] = false
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"
  @assert all(isfinite(pstate.gradlogtarget)) "Gradient of log-target not finite: initial values out of support"
  @assert all(isfinite(pstate.tensorlogtarget)) "Tensor of log-target not finite: initial values out of support"
end

## Initialize GUM state

tuner_state(sampler::GUM, tuner::PSMMALAMCTuner) =
  PSMMALAMCTune(
    BasicMCTune(NaN, 0, 0, tuner.smmalatuner.period),
    BasicMCTune(NaN, 0, 0, tuner.malatuner.period),
    BasicMCTune(sampler.driftstep, 0, 0, tuner.totaltuner.period)
  )

function sampler_state(
  sampler::GUM,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector
)
  sstate = MuvGUMState(generate_empty(pstate), tuner_state(sampler, tuner))
  sstate.sqrttunestep = sqrt(sampler.driftstep)
  sstate.oldinvtensor = inv(pstate.tensorlogtarget)
  sstate.cholinvtensor = chol(sstate.oldinvtensor, Val{:L})
  sstate.oldfirstterm = sstate.oldinvtensor*pstate.gradlogtarget
  sstate.presentupdatetensor = true
  sstate.pastupdatetensor = false
  sstate
end

### Reset GUM sampler

## Reset parameter state

function reset!(
  pstate::ParameterState{Continuous, Multivariate},
  x::RealVector,
  parameter::Parameter{Continuous, Multivariate},
  sampler::GUM
)
  pstate.value = copy(x)
  parameter.uptotensorlogtarget!(pstate)
  pstate.diagnosticvalues[1] = false
end

## Reset sampler state

function reset!(
  sstate::MuvGUMState,
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::GUM,
  tuner::MCTuner
)
  reset!(sstate.tune, sampler, tuner)
  sstate.sqrttunestep = sqrt(sampler.driftstep)
  sstate.oldinvtensor = inv(pstate.tensorlogtarget)
  sstate.cholinvtensor = chol(sstate.oldinvtensor, Val{:L})
  sstate.oldfirstterm = sstate.oldinvtensor*pstate.gradlogtarget
  sstate.presentupdatetensor = true
  sstate.pastupdatetensor = false
end

Base.show(io::IO, sampler::GUM) =
  print(io, "GUM sampler: drift step = $(sampler.driftstep), target rate = $(sampler.targetrate), γ = $(sampler.γ)")

Base.writemime(io::IO, ::MIME"text/plain", sampler::GUM) = show(io, sampler)
