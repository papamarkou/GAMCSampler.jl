### Abstract PGUSMMALA state

abstract PGUSMMALAState <: MCSamplerState

### PGUSMMALA state subtypes

## MuvPGUSMMALAState holds the internal state ("local variables") of the PGUSMMALA sampler for multivariate parameters

type MuvPGUSMMALAState <: PGUSMMALAState
  pstate::ParameterState{Continuous, Multivariate}
  tune::MCTunerState
  newsqrttunestep::Real
  oldsqrttunestep::Real
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
    newsqrttunestep::Real,
    oldsqrttunestep::Real,
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
      @assert newsqrttunestep > 0 "Square root of new tuned drift step is not positive"
      @assert oldsqrttunestep > 0 "Square root of old tuned drift step is not positive"
    end
    new(
      pstate,
      tune,
      newsqrttunestep,
      oldsqrttunestep,
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

exp_decay(i::Integer, tot::Integer, a::Real=10., b::Real=0.) = (1-b)*exp(-a*i/tot)+b

pow_decay(i::Integer, tot::Integer, a::Real=1e-3, b::Real=0.) = (1-b)*(a^(i/tot))+b

lin_decay(i::Integer, tot::Integer, a::Real=1e-3, b::Real=0.) = (1-b)*(1/(1+a*i))+b

quad_decay(i::Integer, tot::Integer, a::Real=1e-3, b::Real=0.) = (1-b)*(1/(1+a*abs2(i)))+b

mala_only_update!(sstate::MuvPGUSMMALAState, pstate::ParameterState{Continuous, Multivariate}, i::Integer, tot::Integer) =
  sstate.presentupdatetensor = false

upto_mala_only_update!(sstate::MuvPGUSMMALAState, pstate::ParameterState{Continuous, Multivariate}, i::Integer, tot::Integer) =
  sstate.presentupdatetensor = i < 3 ? true : false

smmala_only_update!(sstate::MuvPGUSMMALAState, pstate::ParameterState{Continuous, Multivariate}, i::Integer, tot::Integer) =
  sstate.presentupdatetensor = true

mod_update!(
  sstate::MuvPGUSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  n::Integer=10
) =
  sstate.presentupdatetensor = mod(i, n) == 0 ? true : false

cos_update!(
  sstate::MuvPGUSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  a::Real=10.,
  b::Real=0.2,
  c::Real=0.2
) =
  sstate.presentupdatetensor = rand(Bernoulli(b*cos(a*i*pi/tot)+c))

function mahalanobis_update!(
  sstate::MuvPGUSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
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

rand_update!(
  sstate::MuvPGUSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  p::Real=0.5
) =
  sstate.presentupdatetensor = rand(Bernoulli(p))

rand_exp_decay_update!(
  sstate::MuvPGUSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  a::Real=10.,
  b::Real=0.
) =
  sstate.presentupdatetensor = rand(Bernoulli(exp_decay(i, tot, a, b)))

rand_pow_decay_update!(
  sstate::MuvPGUSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  a::Real=1e-3,
  b::Real=0.
) =
  sstate.presentupdatetensor = rand(Bernoulli(pow_decay(i, tot, a, b)))

rand_lin_decay_update!(
  sstate::MuvPGUSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  a::Real=1e-2,
  b::Real=0.
) =
  sstate.presentupdatetensor = rand(Bernoulli(lin_decay(i, tot, a, b)))

rand_quad_decay_update!(
  sstate::MuvPGUSMMALAState,
  pstate::ParameterState{Continuous, Multivariate},
  i::Integer,
  tot::Integer,
  a::Real=1e-5,
  b::Real=0.
) =
  sstate.presentupdatetensor = rand(Bernoulli(quad_decay(i, tot, a, b)))

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
  update::Function=rand_update!,
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
  pstate.diagnosticvalues[1] = false
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
  sstate.newsqrttunestep = sqrt(sstate.tune.step)
  sstate.oldsqrttunestep = sstate.newsqrttunestep
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
  pstate.diagnosticvalues[1] = false
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
  sstate.newsqrttunestep = sqrt(sstate.tune.step)
  sstate.oldsqrttunestep = sstate.newsqrttunestep
  sstate.oldinvtensor = inv(pstate.tensorlogtarget)
  sstate.oldcholinvtensor = chol(sstate.oldinvtensor, Val{:L})
  sstate.oldfirstterm = sstate.oldinvtensor*pstate.gradlogtarget
  sstate.presentupdatetensor, sstate.pastupdatetensor = sampler.initupdatetensor
end

Base.show(io::IO, sampler::PGUSMMALA) = print(io, "PGUSMMALA sampler: drift step = $(sampler.driftstep)")

Base.writemime(io::IO, ::MIME"text/plain", sampler::PGUSMMALA) = show(io, sampler)
