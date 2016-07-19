using Distributions
using Lora
using PGUManifoldMC

plogtarget(p::Vector, v::Vector) = logpdf(MvTDist(30., [1., 2.], [1. 0.; 0. 1.]), p)

p = BasicContMuvParameter(
  :p,
  logtarget=plogtarget,
  nkeys=1,
  autodiff=:forward,
  order=2
)

model = likelihood_model([p], isindexed=false)

# Simulation 01

sampler = PGUSMMALA(
  1.,
  identitymala=false,
  update=(sstate, i, tot) -> rand_exp_decay_update!(sstate, i, tot),
  transform=H -> softabs(H, 1000.),
  initupdatetensor=(true, false)
)

mcrange = BasicMCRange(nsteps=110000, burnin=10000)

v0 = Dict(:p=>[-1., 1.])

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
