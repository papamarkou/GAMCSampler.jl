### AMSMMALA state subtypes

## MuvAMSMMALAState holds the internal state ("local variables") of the AMSMMALA sampler for multivariate parameters

type MuvAMSMMALAState <: MuvPSMMALAState
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

  function MuvAMSMMALAState(
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

MuvAMSMMALAState(pstate::ParameterState{Continuous, Multivariate}, tune::MCTunerState=PSMMALAMCTune()) =
  MuvAMSMMALAState(
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

### Metropolis-adjusted Langevin Algorithm (AMSMMALA)

immutable AMSMMALA <: PSMMALA
  driftstep::Real
  update!::Function
  transform::Union{Function, Void}
  t0::Integer

  function AMSMMALA(driftstep::Real, update!::Function, transform::Union{Function, Void}, t0::Integer)
    @assert driftstep > 0 "Drift step is not positive"
    @assert t0 > 0 "t0 is not positive"
    new(driftstep, update!, transform, t0)
  end
end

AMSMMALA(driftstep::Real=1.; update::Function=rand_update!, transform::Union{Function, Void}=nothing, t0::Integer=3) =
  AMSMMALA(driftstep, update, transform, t0)

### Initialize AMSMMALA sampler

## Initialize parameter state

function initialize!(
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::AMSMMALA,
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

## Initialize AMSMMALA state

tuner_state(sampler::AMSMMALA, tuner::PSMMALAMCTuner) =
  PSMMALAMCTune(
    BasicMCTune(NaN, 0, 0, tuner.smmalatuner.period),
    BasicMCTune(NaN, 0, 0, tuner.malatuner.period),
    BasicMCTune(sampler.driftstep, 0, 0, tuner.totaltuner.period)
  )

function sampler_state(
  parameter::Parameter{Continuous, Multivariate},
  sampler::AMSMMALA,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector
)
  sstate = MuvAMSMMALAState(generate_empty(pstate, parameter.diffmethods, parameter.diffopts), tuner_state(sampler, tuner))
  sstate.sqrttunestep = sqrt(sampler.driftstep)
  sstate.lastmean = copy(pstate.value)
  sstate.oldinvtensor = inv(pstate.tensorlogtarget)
  sstate.cholinvtensor = ctranspose(chol(Hermitian(sstate.oldinvtensor)))
  sstate.oldfirstterm = sstate.oldinvtensor*pstate.gradlogtarget
  sstate.presentupdatetensor = true
  sstate.pastupdatetensor = false
  sstate
end

### Reset AMSMMALA sampler

## Reset parameter state

function reset!(
  pstate::ParameterState{Continuous, Multivariate},
  x::RealVector,
  parameter::Parameter{Continuous, Multivariate},
  sampler::AMSMMALA
)
  pstate.value = copy(x)
  parameter.uptotensorlogtarget!(pstate)
  pstate.diagnosticvalues[1] = false
end

## Reset sampler state

function reset!(
  sstate::MuvAMSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::AMSMMALA,
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

Base.show(io::IO, sampler::AMSMMALA) = print(io, "AMSMMALA sampler: drift step = $(sampler.driftstep)")

Base.show(io::IO, ::MIME"text/plain", sampler::AMSMMALA) = show(io, sampler)
