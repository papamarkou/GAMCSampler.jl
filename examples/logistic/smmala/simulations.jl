using Distributions
using Lora
using PGUManifoldMC

DATADIR = "data"

nchains = 10
nmcmc = 110000
nburnin = 10000

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

pgradlogtarget(p::Vector, v::Vector) = v[2]'*(v[3]-1./(1+exp(-v[2]*p)))-p/v[1]

function ptensorlogtarget(p::Vector, v::Vector)
  r = 1./(1+exp(-v[2]*p))
  (v[2]'.*repmat((r.*(1-r))', length(p), 1))*v[2]+(eye(length(p))/v[1])
end

p = BasicContMuvParameter(
  :p,
  loglikelihood=ploglikelihood,
  logprior=plogprior,
  gradlogtarget=pgradlogtarget,
  tensorlogtarget=ptensorlogtarget,
  nkeys=4
)

model = likelihood_model([Hyperparameter(:λ), Data(:X), Data(:y), p], isindexed=false)

sampler = SMMALA(0.02)

mcrange = BasicMCRange(nsteps=nmcmc, burnin=nburnin)

outopts = Dict{Symbol, Any}(:monitor=>[:value], :diagnostics=>[:accept])

times = Array(Float64, nchains)
stepsizes = Array(Float64, nchains)
i = 1

while i <= nchains
  v0 = Dict(:λ=>100., :X=>covariates, :y=>outcome, :p=>rand(Normal(0, 3), npars))

  job = BasicMCJob(
    model,
    sampler,
    mcrange,
    v0,
    tuner=AcceptanceRateMCTuner(0.7, score=x -> logistic_rate_score(x, 3.), verbose=false),
    outopts=outopts
  )

  tic()
  run(job)
  runtime = toc()

  chain = output(job)
  ratio = acceptance(chain)

  if 0.65 < ratio < 0.75
    writedlm(joinpath(DATADIR, "chain"*lpad(string(i), 2, 0)*".csv"), chain.value, ',')
    writedlm(joinpath(DATADIR, "diagnostics"*lpad(string(i), 2, 0)*".csv"), vec(chain.diagnosticvalues), ',')

    times[i] = runtime
    stepsizes[i] = job.sstate.tune.step

    println("Iteration ", i, " of ", nchains, " completed with acceptance ratio ", ratio)
    i += 1
  end
end

writedlm(joinpath(DATADIR, "times.csv"), times, ',')
writedlm(joinpath(DATADIR, "stepsizes.csv"), stepsizes, ',')
