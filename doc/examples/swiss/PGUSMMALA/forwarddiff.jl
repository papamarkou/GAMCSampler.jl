using Lora
using PGUManifoldMC

covariates, = dataset("swiss", "measurements");
ndata, npars = size(covariates);

covariates = (covariates.-mean(covariates, 1))./repmat(std(covariates, 1), ndata, 1);

outcome, = dataset("swiss", "status");
outcome = vec(outcome);

function ploglikelihood(p::Vector, v::Vector)
  Xp = v[2]*p
  dot(Xp, v[3])-sum(log(1+exp(Xp)))
end

plogprior(p::Vector, v::Vector) = -0.5*(dot(p, p)/v[1]+length(p)*log(2*pi*v[1]))

p = BasicContMuvParameter(
  :p,
  loglikelihood=ploglikelihood,
  logprior=plogprior,
  nkeys=4,
  autodiff=:forward,
  order=2
)

model = likelihood_model([Hyperparameter(:λ), Data(:X), Data(:y), p], isindexed=false)

# sampler = PGUSMMALA(0.02, update=mala_only_update!)

sampler = PGUSMMALA(
  0.02,
  identitymala=false,
  update=(sstate) -> rand_update!(sstate, pstate, 0.3),
  initupdatetensor=(true, false)
)

mcrange = BasicMCRange(nsteps=50000, burnin=10000)

v0 = Dict(:λ=>100., :X=>covariates, :y=>outcome, :p=>[5.1, -0.9, 8.2, -4.5])

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, sampler, mcrange, v0, tuner=AcceptanceRateMCTuner(0.5, verbose=true), outopts=outopts)

run(job)

chain01 = output(job)

mean(chain01)

acceptance(chain01)

# Run a second long chain if you want to estimate ESS

sampler = SMMALA(0.02)

mcrange = BasicMCRange(nsteps=510000, burnin=10000)

v0 = Dict(:λ=>100., :X=>covariates, :y=>outcome, :p=>[5.1, -0.9, 8.2, -4.5])

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, sampler, mcrange, v0, tuner=AcceptanceRateMCTuner(0.5, verbose=true), outopts=outopts)

run(job)

chain02 = output(job)

chain02mean = mean(chain02)

chain02var = Float64[var(vec(chain02.value[i, :])) for i in 1:4]

# Estimate ESS using the mean and (IID) variance of the independent longer chain

Float64[ess(vec(chain01.value[i, :]), chain02mean[i], chain02var[i], chain01.n) for i in 1:4]

ess(chain01, vtype=:bm)
