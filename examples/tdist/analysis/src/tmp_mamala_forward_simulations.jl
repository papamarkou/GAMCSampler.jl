using Distributions
using Klara
using MAMALASampler

DATADIR = "../../data"
SUBDATADIR = "smmala_forward"

nchains = 1
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

plogtarget(p::Vector, v::Vector) = logpdf(MvTDist(ν, zeros(n), (ν-2)*Σ/ν), p)

p = BasicContMuvParameter(
  :p,
  logtarget=plogtarget,
  nkeys=1,
  diffopts=DiffOptions(mode=:forward, order=2)
)

model = likelihood_model([p], isindexed=false)

sampler = MAMALA(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, tot, 10.),
  transform=H -> softabs(H, 1000.),
  driftstep=0.25,
  c=0.001
)

mcrange = BasicMCRange(nsteps=nmcmc, burnin=nburnin)

mctuner = MAMALAMCTuner(
  VanillaMCTuner(verbose=true),
  VanillaMCTuner(verbose=true),
  AcceptanceRateMCTuner(0.27, verbose=true)
)

outopts = Dict{Symbol, Any}(:monitor=>[:value], :diagnostics=>[:accept])

times = Array(Float64, nchains)
stepsizes = Array(Float64, nchains)
i = 1

#while i <= nchains
  v0 = Dict(:p=>rand(Normal(0, 2), n))

  job = BasicMCJob(
    model,
    sampler,
    mcrange,
    v0,
    tuner=mctuner,
    outopts=outopts
  )

  tic()
  run(job)
  runtime = toc()

  chain = output(job)
  ratio = acceptance(chain)

  mean(chain)
  ess(chain)

  if 0.15 < ratio < 0.35
    # writedlm(joinpath(DATADIR, SUBDATADIR, "chain"*lpad(string(i), 2, 0)*".csv"), chain.value, ',')
    #writedlm(joinpath(DATADIR, SUBDATADIR, "diagnostics"*lpad(string(i), 2, 0)*".csv"), vec(chain.diagnosticvalues), ',')

    times[i] = runtime
    stepsizes[i] = job.sstate.tune.totaltune.step

    println("Iteration ", i, " of ", nchains, " completed with acceptance ratio ", ratio)
    # i += 1
  end
# end

# writedlm(joinpath(DATADIR, SUBDATADIR, "times.csv"), times, ',')
# writedlm(joinpath(DATADIR, SUBDATADIR, "stepsizes.csv"), stepsizes, ',')
