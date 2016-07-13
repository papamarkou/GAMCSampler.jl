using Distributions
using Lora
using PGUManifoldMC

pmean = [4.1, 3.7]
pvar = [1., 1.]

plogtarget(p::Vector, v::Vector) = logpdf(MvNormal(pmean, [pvar[1] 0.75; 0.75 pvar[2]]), p)

p = BasicContMuvParameter(
  :p,
  logtarget=plogtarget,
  nkeys=1,
  autodiff=:forward,
  order=2
)

model = likelihood_model([p], isindexed=false)

# Simulation 01

sampler = MALA(0.02)

mcrange = BasicMCRange(nsteps=50000, burnin=10000)

v0 = Dict(:p=>[-9.1, -0.9])

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, sampler, mcrange, v0, tuner=AcceptanceRateMCTuner(0.5, verbose=true), outopts=outopts)

run(job)

chain = output(job)

mean(chain)

ess(chain, vtype=:bm)

Float64[ess(vec(chain.value[i, :]), pmean[i], pvar[i], chain.n) for i in 1:2]

acceptance(chain)

# Simulation 02

sampler = SMMALA(0.02)

mcrange = BasicMCRange(nsteps=50000, burnin=10000)

v0 = Dict(:p=>[-9.1, -0.9])

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, sampler, mcrange, v0, tuner=AcceptanceRateMCTuner(0.5, verbose=true), outopts=outopts)

run(job)

chain = output(job)

mean(chain)

ess(chain, vtype=:bm)

Float64[ess(vec(chain.value[i, :]), pmean[i], pvar[i], chain.n) for i in 1:2]

acceptance(chain)

# Simulation 03

sampler = PGUSMMALA(
  0.02,
  identitymala=false,
  update=(sstate) -> rand_update!(sstate, 0.3),
  initupdatetensor=(true, false)
)

mcrange = BasicMCRange(nsteps=50000, burnin=10000)

v0 = Dict(:p=>[-9.1, -0.9])

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, sampler, mcrange, v0, tuner=AcceptanceRateMCTuner(0.5, verbose=true), outopts=outopts)

run(job)

chain = output(job)

mean(chain)

ess(chain, vtype=:bm)

Float64[ess(vec(chain.value[i, :]), pmean[i], pvar[i], chain.n) for i in 1:2]

acceptance(chain)

round(100*job.sstate.updatetensorcount/job.range.nsteps, 2)

# Simulation 04

sampler = PGUSMMALA(
  0.02,
  identitymala=true,
  update=(sstate) -> rand_update!(sstate, 0.3),
  initupdatetensor=(true, false)
)

mcrange = BasicMCRange(nsteps=50000, burnin=10000)

v0 = Dict(:p=>[-9.1, -0.9])

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget], :diagnostics=>[:accept])

job = BasicMCJob(model, sampler, mcrange, v0, tuner=AcceptanceRateMCTuner(0.5, verbose=true), outopts=outopts)

run(job)

chain = output(job)

mean(chain)

ess(chain, vtype=:bm)

Float64[ess(vec(chain.value[i, :]), pmean[i], pvar[i], chain.n) for i in 1:2]

acceptance(chain)

round(100*job.sstate.updatetensorcount/job.range.nsteps, 2)
