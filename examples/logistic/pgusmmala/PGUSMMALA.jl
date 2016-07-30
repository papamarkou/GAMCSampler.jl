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

sampler = PGUSMMALA(
  0.02,
  identitymala=false,
  update=(sstate, i, tot) -> rand_exp_decay_update!(sstate, i, tot),
  initupdatetensor=(true, false)
)

mcrange = BasicMCRange(nsteps=50000, burnin=10000)

v0 = Dict(:λ=>100., :X=>covariates, :y=>outcome, :p=>[5.1, -0.9, 8.2, -4.5])

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(
  model,
  sampler,
  mcrange,
  v0,
  tuner=AcceptanceRateMCTuner(0.6, score=x -> logistic_rate_score(x, 3.), verbose=false),
  outopts=outopts
)

tic()
run(job)
runtime = toc()

chain = output(job)

ppostmean = mean(chain)

ess(chain, vtype=:bm)

ess(chain, vtype=:bm)/runtime

acceptance(chain)
