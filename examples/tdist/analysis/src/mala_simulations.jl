using Distributions
# using Gadfly
using Lora
using PGUManifoldMC

function C(n::Int, c::Float64)
  X = eye(n)
  [(j <= n-i) ? X[i+j, i] = X[i, i+j] = c^j : nothing for i = 1:(n-1), j = 1:(n-1)]
  X
end

n = 20
μ = zeros(n)
Σ = C(n, 0.9)
ν = 30.

Σt = (ν-2)*Σ/ν
Σtinv = inv(Σt)

function plogtarget(p::Vector, v::Vector)
  hdf = 0.5*ν
  hdim = 0.5*n
  shdfhdim = hdf+hdim
  v = lgamma(shdfhdim)-lgamma(hdf)-hdim*log(ν)-hdim*log(pi)-0.5*logdet(Σt)
  z = p-μ
  v-shdfhdim*log1p(dot(z, Σtinv*z)/ν)
end

v0 = Dict(:p=>[-4., 2., 3., 1., 2.4, -4., 2., 3., 1., 2.4, -4., 2., 3., 1., 2.4, -4., 2., 3., 1., 2.4])

p = BasicContMuvParameter(
  :p,
  logtarget=plogtarget,
  nkeys=1,
  autodiff=:reverse,
  init=Any[(:p, v0[:p]), (:v, Any[v0[:p]])],
  order=1
)

model = likelihood_model([p], isindexed=false)

sampler = MALA(0.05)

mcrange = BasicMCRange(nsteps=110000, burnin=10000)

outopts = Dict{Symbol, Any}(:monitor=>[:value], :diagnostics=>[:accept])

job = BasicMCJob(
  model,
  sampler,
  mcrange,
  v0,
  tuner=AcceptanceRateMCTuner(0.574, score=x -> logistic_rate_score(x, 3.), verbose=false),
  outopts=outopts
)

tic()
run(job)
runtime = toc()

chain = output(job)

ppostmean = mean(chain)

essizes = ess(chain)

essizes/runtime

acceptance(chain)

# plot(x=collect(1:100000), y=chain.value[1, :], Geom.line)

# plot(x=collect(1:100000), y=[mean(chain.value[1, 1:i]) for i in 1:100000], Geom.line)
