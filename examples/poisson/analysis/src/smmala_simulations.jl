using Distributions
using Lora
using PGUManifoldMC

DATADIR = "../../data"
SUBDATADIR = "smmala"

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
  dot(Xp, v[3])-sum(exp(Xp))-sum(lfact(v[3]))
end

plogprior(p::Vector{Float64}, v::Vector) = -0.5*(dot(p, p)/v[1]+npars*log(2*pi*v[1]))

pgradlogtarget(p::Vector{Float64}, v::Vector) = v[2]'*(v[3]-exp(v[2]*p))-p/v[1]

function ptensorlogtarget(p::Vector{Float64}, v::Vector)
  broadcast(*, exp(v[2]*p), v[2])'*v[2]+(eye(npars)/v[1])
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

sampler = SMMALA(0.02, softabs)

mcrange = BasicMCRange(nsteps=nmcmc, burnin=nburnin)

outopts = Dict{Symbol, Any}(:monitor=>[:value], :diagnostics=>[:accept])

times = Array(Float64, nchains)
stepsizes = Array(Float64, nchains)
i = 1

v0 = Dict(:λ=>100., :X=>covariates, :y=>outcome, :p=>[1.1, 0.9, 1.2, 4.5])

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

ppostmean = mean(chain)

ess(chain, vtype=:bm)

ess(chain, vtype=:bm)/runtime

acceptance(chain)
