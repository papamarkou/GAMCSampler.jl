using Distributions
using Lora
using PGUManifoldMC

n = 100
ystd = 2.
pmean = [0., 0.]
pvar = [1., 1.]
ptrue = [0.9, 0.1]

y = rand(Normal(ptrue[1]+abs2(ptrue[2]), ystd), 100)

ploglikelihood(p::Vector, v::Vector) = sum([logpdf(Normal(p[1]+abs2(p[2]), ystd), v[1][i]) for i in 1:n])

plogprior(p::Vector, v::Vector) = logpdf(MvNormal(pmean, pvar), p)

p = BasicContMuvParameter(
  :p,
  loglikelihood=ploglikelihood,
  logprior=plogprior,
  nkeys=2,
  autodiff=:forward,
  order=2
)

model = likelihood_model([Data(:y), p], isindexed=false)

# Simulation 01

sampler = SMMALA(1., softabs)

# sampler = PGUSMMALA(
#   1,
#   identitymala=false,
#   update=(sstate) -> rand_update!(sstate, 0.3),
#   transform=softabs,
#   initupdatetensor=(true, false)
# )

mcrange = BasicMCRange(nsteps=11000, burnin=1000)

v0 = Dict(:y=>y, :p=>[-6., 5.])

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(
  model,
  sampler,
  mcrange,
  v0,
  tuner=AcceptanceRateMCTuner(0.5, score=x -> logistic_rate_score(x, 3.), verbose=false),
  outopts=outopts
)

run(job)

chain = output(job)

ppostmean = mean(chain)

ppostmean-ptrue

ess(chain, vtype=:bm)

# Float64[ess(vec(chain.value[i, :]), θ[i], pvar[i], chain.n) for i in 1:n]

acceptance(chain)
