using Distributions
using Lora
using PGUManifoldMC

n = 2

# θ = [0., 0.]
# pvar = [10., 1.]

θ = [0., 0.]
pvar = [9., 1.]
b = 0.015

# plogtarget(p::Vector, v::Vector) =
  # logpdf(MvNormal(θ, pvar), [p[1], p[2]+0.03*abs2(p[1])-0.3])
  # logpdf(MvNormal(θ, pvar), [p[1], p[2]+0.02*(abs2(p[1])-pvar[1])])

plogtarget(p::Vector, v::Vector) =
  # logpdf(MvNormal(θ, pvar), [p[1], p[2]+0.03*abs2(p[1])-0.3])
  logpdf(MvNormal(θ, pvar), [p[1], p[2]+b*(abs2(p[1])-pvar[1])])

p = BasicContMuvParameter(
  :p,
  logtarget=plogtarget,
  nkeys=1,
  autodiff=:forward,
  order=2
)

model = likelihood_model([p], isindexed=false)

# Simulation 01

sampler = SMMALA(1., softabs)

# sampler = PGUSMMALA(
#   1,
#   identitymala=false,
#   update=(sstate) -> rand_update!(sstate, 0.3),
#   transform=softabs,
#   initupdatetensor=(true, false)
# )

mcrange = BasicMCRange(nsteps=110000, burnin=10000)

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
