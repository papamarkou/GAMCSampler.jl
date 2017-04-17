### MAMALA state subtypes

abstract MAMALAState <: LMCSamplerState

## MuvMAMALAState holds the internal state ("local variables") of the MAMALA sampler for multivariate parameters

type MuvMAMALAState <: MAMALAState
  proposal::MultivariateGMM
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
  w::RealVector
  count::Integer
  updatetensorcount::Integer

  function MuvMAMALAState(
    proposal::MultivariateGMM,
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
    w::RealVector,
    count::Integer,
    updatetensorcount::Integer
  )
    if !isnan(ratio)
      @assert 0 < ratio < 1 "Acceptance ratio should be between 0 and 1"
      @assert sqrttunestep > 0 "Square root of tuned drift step is not positive"
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
      presentupdatetensor,
      pastupdatetensor,
      w,
      count,
      updatetensorcount
    )
  end
end

MuvMAMALAState(
  proposal::MultivariateGMM,
  pstate::ParameterState{Continuous, Multivariate},
  w::RealVector,
  tune::MCTunerState=MAMALAMCTune()
) =
  MuvMAMALAState(
  proposal,
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
  w,
  0,
  0
)

### Manifold adaptive Metropolis-adjusted Langevin algorithm (MAMALA)

immutable MAMALA <: LMCSampler
  update!::Function
  transform::Union{Function, Void}
  driftstep::Real
  corescale::Real # Scaling factor of covariance matrix of the core mixture component
  minorscale::Real # Scaling factor of covariance matrix of the stabilizing mixture component
  c::Real # Non-negative constant with relative small value that determines the mixture weight of the stabilizing component
  t0::Integer

  function MAMALA(
    update!::Function,
    transform::Union{Function, Void},
    driftstep::Real,
    corescale::Real,
    minorscale::Real,
    c::Real,
    t0::Integer
  )
    @assert driftstep > 0 "Drift step is not positive"
    @assert 0 < corescale "Constant corescale must be positive, got $corescale"
    @assert 0 < minorscale "Constant minorscale must be positive, got $minorscale"
    @assert 0 <= c "Constant c must be non-negative"
    @assert t0 > 0 "t0 is not positive"
    new(update!, transform, driftstep, corescale, minorscale, c, t0)
  end
end

MAMALA(;
  update::Function=rand_update!,
  transform::Union{Function, Void}=nothing,
  driftstep::Real=1.,
  corescale::Real=1.,
  minorscale::Real=1.,
  c::Real=0.05,
  t0::Integer=3
) =
  MAMALA(update, transform, driftstep, corescale, minorscale, c, t0)

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

setproposal(sampler::MAMALA, pstate::ParameterState{Continuous, Multivariate}, C::RealMatrix, w::RealVector) =
  MixtureModel([MvNormal(pstate.value, sampler.corescale*C), MvNormal(pstate.value, sampler.minorscale*eye(pstate.size))], w)

setproposal!(sstate::MuvMAMALAState, sampler::MAMALA, pstate::ParameterState{Continuous, Multivariate}) =
  sstate.proposal = setproposal(sampler, pstate, sstate.oldinvtensor, sstate.w)

tuner_state(sampler::MAMALA, tuner::MAMALAMCTuner) =
  MAMALAMCTune(
    BasicMCTune(NaN, 0, 0, tuner.smmalatuner.period),
    BasicMCTune(NaN, 0, 0, tuner.amtuner.period),
    BasicMCTune(sampler.driftstep, 0, 0, tuner.totaltuner.period)
  )

MuvMAMALAState(
  proposal::MultivariateGMM,
  pstate::ParameterState{Continuous, Multivariate},
  w::RealVector,
  tune::MCTunerState=MAMALAMCTune()
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
    setproposal(sampler, pstate, oldinvtensor, w),
    generate_empty(pstate, parameter.diffmethods, parameter.diffopts),
    w,
    tuner_state(sampler, tuner)
  )

  sstate.sqrttunestep = sqrt(sampler.driftstep)
  sstate.lastmean = copy(pstate.value)
  sstate.oldinvtensor = inv(pstate.tensorlogtarget)
  # sstate.cholinvtensor = ctranspose(chol(Hermitian(sstate.oldinvtensor)))
  # sstate.oldfirstterm = sstate.oldinvtensor*pstate.gradlogtarget
  # sstate.presentupdatetensor = true
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
  # sstate.cholinvtensor = chol(sstate.oldinvtensor, Val{:L})
  # sstate.oldfirstterm = sstate.oldinvtensor*pstate.gradlogtarget
  # sstate.presentupdatetensor = true
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
