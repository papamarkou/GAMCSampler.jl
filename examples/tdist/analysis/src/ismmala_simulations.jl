using Distributions
using Lora
using PGUManifoldMC

DATADIR = "../../data"
SUBDATADIR = "ismmala"

nchains = 10
nmcmc = 110000
nburnin = 10000

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

function plogtarget(p::Vector{Float64}, v::Vector)
  hdf = 0.5*ν
  hdim = 0.5*n
  shdfhdim = hdf+hdim
  v = lgamma(shdfhdim)-lgamma(hdf)-hdim*log(ν)-hdim*log(pi)-0.5*logdet(Σt)
  z = p-μ
  v-shdfhdim*log1p(dot(z, Σtinv*z)/ν)
end

sampler = ISMMALA(
  0.5,
  identitymala=false,
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, tot, 7., 0.05),
  transform=H -> softabs(H, 1000.),
  initupdatetensor=(true, false)
)

mcrange = BasicMCRange(nsteps=nmcmc, burnin=nburnin)

outopts = Dict{Symbol, Any}(:monitor=>[:value], :diagnostics=>[:accept])

times = Array(Float64, nchains)
stepsizes = Array(Float64, nchains)
nupdates = Array(Int64, nchains)
i = 1

while i <= nchains
  v0 = Dict(:p=>rand(Normal(0, 2), n))

  p = BasicContMuvParameter(
    :p,
    logtarget=plogtarget,
    nkeys=1,
    autodiff=:reverse,
    init=Any[(:p, v0[:p]), (:v, Any[v0[:p]])],
    order=2
  )

  model = likelihood_model([p], isindexed=false)

  job = BasicMCJob(
    model,
    sampler,
    mcrange,
    v0,
    tuner=VanillaMCTuner(verbose=false),
    outopts=outopts
  )

  tic()
  run(job)
  runtime = toc()

  chain = output(job)
  ratio = acceptance(chain)

  if 0.85 < ratio < 0.92
    writedlm(joinpath(DATADIR, SUBDATADIR, "chain"*lpad(string(i), 2, 0)*".csv"), chain.value, ',')
    writedlm(joinpath(DATADIR, SUBDATADIR, "diagnostics"*lpad(string(i), 2, 0)*".csv"), vec(chain.diagnosticvalues), ',')

    times[i] = runtime
    stepsizes[i] = job.sstate.tune.step
    nupdates[i] = job.sstate.updatetensorcount

    println("Iteration ", i, " of ", nchains, " completed with acceptance ratio ", ratio)
    i += 1
  else
    println("Iteration ", i, " of ", nchains, " ignored with acceptance ratio ", ratio)
  end
end

writedlm(joinpath(DATADIR, SUBDATADIR, "times.csv"), times, ',')
writedlm(joinpath(DATADIR, SUBDATADIR, "stepsizes.csv"), stepsizes, ',')
writedlm(joinpath(DATADIR, SUBDATADIR, "nupdates.csv"), nupdates, ',')
