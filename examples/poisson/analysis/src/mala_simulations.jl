using Distributions
using Lora
using PGUManifoldMC

DATADIR = "../../data"
SUBDATADIR = "mala"

nchains = 10
nmcmc = 110000
nburnin = 10000

dataset, = readdlm(joinpath(DATADIR, "coarse_bei.csv"), ',', header=true);

covariates = dataset[:, 2:3];
ndata, npars = size(covariates);
npars += 2

covariates = (covariates.-mean(covariates, 1))./repmat(std(covariates, 1), ndata, 1);
covariates = [ones(200) covariates[:, 1] map(abs2, covariates[:, 1]) covariates[:, 2]];

outcome = dataset[:, 1];

function ploglikelihood(p::Vector{Float64}, v::Vector)
  Xp = v[2]*p
  # dot(Xp, v[3])-sum(exp(Xp))-sum(lfact(v[3]))
  dot(Xp, v[3])-sum(exp(Xp))
end

plogprior(p::Vector{Float64}, v::Vector) = -0.5*(dot(p, p)/v[1]+npars*log(2*pi*v[1]))

pgradlogtarget(p::Vector{Float64}, v::Vector) = v[2]'*(v[3]-exp(v[2]*p))-p/v[1]

p = BasicContMuvParameter(
  :p,
  loglikelihood=ploglikelihood,
  logprior=plogprior,
  nkeys=4,
  gradlogtarget=pgradlogtarget,
  order=1
)

model = likelihood_model([Hyperparameter(:λ), Data(:X), Data(:y), p], isindexed=false)

sampler = MALA(0.02)

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
    tuner=AcceptanceRateMCTuner(0.574, score=x -> logistic_rate_score(x, 3.), verbose=false),
    outopts=outopts
  )

  tic()
  run(job)
  runtime = toc()

  chain = output(job)
  ratio = acceptance(chain)

  if 0.5 < ratio < 0.65
    writedlm(joinpath(DATADIR, SUBDATADIR, "chain"*lpad(string(i), 2, 0)*".csv"), chain.value, ',')
    writedlm(joinpath(DATADIR, SUBDATADIR, "diagnostics"*lpad(string(i), 2, 0)*".csv"), vec(chain.diagnosticvalues), ',')

    times[i] = runtime
    stepsizes[i] = job.sstate.tune.step

    println("Iteration ", i, " of ", nchains, " completed with acceptance ratio ", ratio)
    i += 1
  end
end

writedlm(joinpath(DATADIR, SUBDATADIR, "times.csv"), times, ',')
writedlm(joinpath(DATADIR, SUBDATADIR, "stepsizes.csv"), stepsizes, ',')
