using Distributions
using Lora
using PGUManifoldMC

n = 7
pmean = [4.1, 3.7, -9.1, -5.4, 2.1, 5.4, 6.5]
pvar = ones(7)
S = diagm(pvar)

plogtarget(p::Vector, v::Vector) = logpdf(MvNormal(pmean, S), p)

p = BasicContMuvParameter(
  :p,
  logtarget=plogtarget,
  nkeys=1,
  autodiff=:forward,
  order=2
)

model = likelihood_model([p], isindexed=false)

# Simulation 01

sampler = MALA(1.5)

mcrange = BasicMCRange(nsteps=50000, burnin=10000)

v0 = Dict(:p=>[-4., -3, 1.1, 4.5, 11., 2., 1.5])

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

mean(chain)-pmean

ess(chain, vtype=:bm)

Float64[ess(vec(chain.value[i, :]), pmean[i], pvar[i], chain.n) for i in 1:n]

acceptance(chain)

# Simulation 02

sampler = SMMALA(1.5)

mcrange = BasicMCRange(nsteps=50000, burnin=10000)

v0 = Dict(:p=>[-4., -3, 1.1, 4.5, 11., 2., 1.5])

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, sampler, mcrange, v0, tuner=AcceptanceRateMCTuner(0.5, verbose=false), outopts=outopts)

run(job)

chain = output(job)

mean(chain)

mean(chain)-pmean

ess(chain, vtype=:bm)

Float64[ess(vec(chain.value[i, :]), pmean[i], pvar[i], chain.n) for i in 1:n]

acceptance(chain)

# Simulation 03

sampler = PGUSMMALA(
  1.5,
  identitymala=false,
  update=(sstate) -> rand_update!(sstate, 0.1),
  initupdatetensor=(true, false)
)

mcrange = BasicMCRange(nsteps=50000, burnin=10000)

v0 = Dict(:p=>[-4., -3, 1.1, 4.5, 11., 2., 1.5])

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, sampler, mcrange, v0, tuner=AcceptanceRateMCTuner(0.5, verbose=false), outopts=outopts)

run(job)

chain = output(job)

mean(chain)

mean(chain)-pmean

ess(chain, vtype=:bm)

Float64[ess(vec(chain.value[i, :]), pmean[i], pvar[i], chain.n) for i in 1:n]

acceptance(chain)

round(100*job.sstate.updatetensorcount/job.range.nsteps, 2)

# Simulation 04

sampler = PGUSMMALA(
  1.5,
  identitymala=true,
  update=(sstate) -> rand_update!(sstate, 0.1),
  initupdatetensor=(true, false)
)

mcrange = BasicMCRange(nsteps=50000, burnin=10000)

v0 = Dict(:p=>[-4., -3, 1.1, 4.5, 11., 2., 1.5])

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, sampler, mcrange, v0, tuner=AcceptanceRateMCTuner(0.5, verbose=false), outopts=outopts)

run(job)

chain = output(job)

mean(chain)

mean(chain)-pmean

ess(chain, vtype=:bm)

Float64[ess(vec(chain.value[i, :]), pmean[i], pvar[i], chain.n) for i in 1:n]

acceptance(chain)

round(100*job.sstate.updatetensorcount/job.range.nsteps, 2)
