using Distributions
using Lora
using PGUManifoldMC

plogtarget(p::Vector, v::Vector) =
  log(0.5*(
    pdf(MvNormal([2., 5.], [1 0.95; 0.95 1]), p)+
    pdf(MvNormal([3., 6.], [1 -0.95; -0.95 1]), p)
  ))

p = BasicContMuvParameter(
  :p,
  logtarget=plogtarget,
  nkeys=1,
  autodiff=:forward,
  order=2
)

model = likelihood_model([p], isindexed=false)

# Simulation 01

# sampler = SMMALA(0.02, softabs)

sampler = PGUSMMALA(
  0.01,
  identitymala=false,
  update=(sstate) -> rand_update!(sstate, 0.8),
  transform=softabs,
  initupdatetensor=(true, false)
)

mcrange = BasicMCRange(nsteps=11000, burnin=1000)

v0 = Dict(:p=>[-6., 5.])

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

mean(chain)

ess(chain, vtype=:bm)

# Float64[ess(vec(chain.value[i, :]), θ[i], pvar[i], chain.n) for i in 1:n]

acceptance(chain)
