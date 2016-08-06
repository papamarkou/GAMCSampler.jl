### Abstract ASMMALA state

abstract ASMMALAState <: MCSamplerState

### ASMMALA state subtypes

## MuvASMMALAState holds the internal state ("local variables") of the ASMMALA sampler for multivariate parameters

type MuvASMMALAState <: ASMMALAState
  pstate::ParameterState{Continuous, Multivariate}
  p0value::RealVector
  tune::MCTunerState
  sqrttunestep::Real
  ratio::Real
  μ::RealVector
  newinvtensor::RealMatrix
  oldinvtensor::RealMatrix
  newcholinvtensor::RealLowerTriangular
  oldcholinvtensor::RealLowerTriangular
  newfirstterm::RealVector
  oldfirstterm::RealVector
  presentupdatetensor::Bool
  pastupdatetensor::Bool
  count::Integer
  updatetensorcount::Integer

  function MuvASMMALAState(
    pstate::ParameterState{Continuous, Multivariate},
    p0value::RealVector,
    tune::MCTunerState,
    sqrttunestep::Real,
    ratio::Real,
    μ::RealVector,
    newinvtensor::RealMatrix,
    oldinvtensor::RealMatrix,
    newcholinvtensor::RealLowerTriangular,
    oldcholinvtensor::RealLowerTriangular,
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
      p0value,
      tune,
      sqrttunestep,
      ratio,
      μ,
      newinvtensor,
      oldinvtensor,
      newcholinvtensor,
      oldcholinvtensor,
      newfirstterm,
      oldfirstterm,
      presentupdatetensor,
      pastupdatetensor,
      count,
      updatetensorcount
    )
  end
end

MuvASMMALAState(pstate::ParameterState{Continuous, Multivariate}, tune::MCTunerState=BasicMCTune()) =
  MuvASMMALAState(
  pstate,
  pstate.value,
  tune,
  NaN,
  NaN,
  Array(eltype(pstate), pstate.size),
  Array(eltype(pstate), pstate.size, pstate.size),
  Array(eltype(pstate), pstate.size, pstate.size),
  RealLowerTriangular(Array(eltype(pstate), pstate.size, pstate.size)),
  RealLowerTriangular(Array(eltype(pstate), pstate.size, pstate.size)),
  Array(eltype(pstate), pstate.size),
  Array(eltype(pstate), pstate.size),
  false,
  false,
  0,
  0
)

immutable ASMMALA <: LMCSampler
  driftstep::Real
  identitymala::Bool
  update!::Function
  transform::Union{Function, Void}
  initupdatetensor::Tuple{Bool,Bool} # The tuple ordinates refer to (sstate.presentupdatetensor, sstate.pastupdatetensor)

  function ASMMALA(
    driftstep::Real,
    identitymala::Bool,
    update!::Function,
    transform::Union{Function, Void},
    initupdatetensor::Tuple{Bool,Bool}
    )
    @assert driftstep > 0 "Drift step is not positive"
    new(driftstep, identitymala, update!, transform, initupdatetensor)
  end
end

mahalanobis_update!(sstate::MuvASMMALAState, pstate::ParameterState{Continuous, Multivariate}, a::Real=0.95) =
  sstate.presentupdatetensor =
    !in_mahalanobis_contour(sstate.pstate.value, sstate.p0value, pstate.tensorlogtarget/sstate.tune.step, pstate.size)

ASMMALA(
  driftstep::Real=1.;
  identitymala::Bool=false,
  update::Function=rand_update!,
  transform::Union{Function, Void}=nothing,
  initupdatetensor::Tuple{Bool,Bool}=(false, false)
) =
  ASMMALA(driftstep, identitymala, update, transform, initupdatetensor)

### Initialize ASMMALA sampler

## Initialize parameter state

function initialize!(
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::ASMMALA
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

## Initialize ASMMALA state

function sampler_state(
  sampler::ASMMALA,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector
)
  sstate = MuvASMMALAState(generate_empty(pstate), tuner_state(sampler, tuner))
  sstate.sqrttunestep = sqrt(sstate.tune.step)
  sstate.oldinvtensor = inv(pstate.tensorlogtarget)
  sstate.oldcholinvtensor = sstate.sqrttunestep*chol(sstate.oldinvtensor, Val{:L})
  sstate.oldfirstterm = sstate.oldinvtensor*pstate.gradlogtarget
  sstate.presentupdatetensor, sstate.pastupdatetensor = sampler.initupdatetensor
  sstate
end

### Reset ASMMALA sampler

## Reset parameter state

function reset!(
  pstate::ParameterState{Continuous, Multivariate},
  x::RealVector,
  parameter::Parameter{Continuous, Multivariate},
  sampler::ASMMALA
)
  pstate.value = copy(x)
  parameter.uptotensorlogtarget!(pstate)
  pstate.diagnosticvalues[1] = false
end

## Reset sampler state

function reset!(
  sstate::ASMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::ASMMALA,
  tuner::MCTuner
)
  reset!(sstate.tune, sampler, tuner)
  sstate.p0value = pstate.value
  sstate.sqrttunestep = sqrt(sstate.tune.step)
  sstate.oldinvtensor = inv(pstate.tensorlogtarget)
  sstate.oldcholinvtensor = sstate.sqrttunestep*chol(sstate.oldinvtensor, Val{:L})
  sstate.oldfirstterm = sstate.oldinvtensor*pstate.gradlogtarget
  sstate.presentupdatetensor, sstate.pastupdatetensor = sampler.initupdatetensor
end

Base.show(io::IO, sampler::ASMMALA) = print(io, "ASMMALA sampler: drift step = $(sampler.driftstep)")

Base.writemime(io::IO, ::MIME"text/plain", sampler::ASMMALA) = show(io, sampler)
