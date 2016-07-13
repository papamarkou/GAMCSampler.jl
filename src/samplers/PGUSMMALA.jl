### Abstract PGUSMMALA state

abstract PGUSMMALAState <: MCSamplerState

### PGUSMMALA state subtypes

## MuvPGUSMMALAState holds the internal state ("local variables") of the PGUSMMALA sampler for multivariate parameters

type MuvPGUSMMALAState <: PGUSMMALAState
  pstate::ParameterState{Continuous, Multivariate}
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
  updatetensorcount::Int

  function MuvPGUSMMALAState(
    pstate::ParameterState{Continuous, Multivariate},
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
    updatetensorcount::Int
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
      newcholinvtensor,
      oldcholinvtensor,
      newfirstterm,
      oldfirstterm,
      presentupdatetensor,
      pastupdatetensor,
      updatetensorcount
    )
  end
end

MuvPGUSMMALAState(pstate::ParameterState{Continuous, Multivariate}, tune::MCTunerState=BasicMCTune()) =
  MuvPGUSMMALAState(
  pstate,
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
  0
)

mala_only_update!(sstate::MuvPGUSMMALAState, pstate::ParameterState{Continuous, Multivariate}) =
  sstate.presentupdatetensor = false

smmala_only_update!(sstate::MuvPGUSMMALAState, pstate::ParameterState{Continuous, Multivariate}) =
  sstate.presentupdatetensor = true

rand_update!(sstate::MuvPGUSMMALAState, pstate::ParameterState{Continuous, Multivariate}, p::Real=0.5) =
  sstate.presentupdatetensor = rand(Bernoulli(p))

function mahalanobis_update!(
  sstate::MuvPGUSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  a::Real=0.95
)
  sstate.presentupdatetensor =
    if dot(
      sstate.pstate.value-pstate.value,
      pstate.tensorlogtarget*(sstate.pstate.value-pstate.value)
    )/sstate.tune.step > quantile(Chisq(pstate.size), a)
      true
    else
      false
    end
end

### Metropolis-adjusted Langevin Algorithm (PGUSMMALA)

immutable PGUSMMALA <: LMCSampler
  driftstep::Real
  identitymala::Bool
  update!::Function
  transform::Union{Function, Void}
  initupdatetensor::Tuple{Bool,Bool} # The tuple ordinates refer to (sstate.presentupdatetensor, sstate.pastupdatetensor)

  function PGUSMMALA(
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

PGUSMMALA(
  driftstep::Real=1.;
  identitymala::Bool=false,
  update::Function=mahalanobis_update!,
  transform::Union{Function, Void}=nothing,
  initupdatetensor::Tuple{Bool,Bool}=(false, false)
) =
  PGUSMMALA(driftstep, identitymala, update, transform, initupdatetensor)

### Initialize PGUSMMALA sampler

## Initialize parameter state

function initialize!(
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::PGUSMMALA
)
  parameter.uptotensorlogtarget!(pstate)
  if sampler.transform != nothing
    pstate.tensorlogtarget = sampler.transform(pstate.tensorlogtarget)
  end
  @assert isfinite(pstate.logtarget) "Log-target not finite: initial value out of support"
  @assert all(isfinite(pstate.gradlogtarget)) "Gradient of log-target not finite: initial values out of support"
  @assert all(isfinite(pstate.tensorlogtarget)) "Tensor of log-target not finite: initial values out of support"
end

## Initialize PGUSMMALA state

function sampler_state(
  sampler::PGUSMMALA,
  tuner::MCTuner,
  pstate::ParameterState{Continuous, Multivariate},
  vstate::VariableStateVector
)
  sstate = MuvPGUSMMALAState(generate_empty(pstate), tuner_state(sampler, tuner))
  sstate.sqrttunestep = sqrt(sstate.tune.step)
  sstate.oldinvtensor = inv(pstate.tensorlogtarget)
  sstate.oldcholinvtensor = chol(sstate.oldinvtensor, Val{:L})
  sstate.oldfirstterm = sstate.oldinvtensor*pstate.gradlogtarget
  sstate.presentupdatetensor, sstate.pastupdatetensor = sampler.initupdatetensor
  sstate
end

### Reset PGUSMMALA sampler

## Reset parameter state

function reset!(
  pstate::ParameterState{Continuous, Multivariate},
  x::RealVector,
  parameter::Parameter{Continuous, Multivariate},
  sampler::PGUSMMALA
)
  pstate.value = copy(x)
  parameter.uptotensorlogtarget!(pstate)
end

## Reset sampler state

function reset!(
  sstate::PGUSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  parameter::Parameter{Continuous, Multivariate},
  sampler::PGUSMMALA,
  tuner::MCTuner
)
  reset!(sstate.tune, sampler, tuner)
  sstate.sqrttunestep = sqrt(sstate.tune.step)
  sstate.oldinvtensor = inv(pstate.tensorlogtarget)
  sstate.oldcholinvtensor = chol(sstate.oldinvtensor, Val{:L})
  sstate.oldfirstterm = sstate.oldinvtensor*pstate.gradlogtarget
  sstate.presentupdatetensor, sstate.pastupdatetensor = sampler.initupdatetensor
end

Base.show(io::IO, sampler::PGUSMMALA) = print(io, "PGUSMMALA sampler: drift step = $(sampler.driftstep)")

Base.writemime(io::IO, ::MIME"text/plain", sampler::PGUSMMALA) = show(io, sampler)
