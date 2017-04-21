using Distributions
using Klara
using MAMALASampler

nmcmc = 110000
nburnin = 10000

covariates, = dataset("swiss", "measurements");
ndata, npars = size(covariates);

covariates = (covariates.-mean(covariates, 1))./repmat(std(covariates, 1), ndata, 1);

outcome, = dataset("swiss", "status");
outcome = vec(outcome);

function ploglikelihood(p::Vector{Float64}, v::Vector)
  Xp = v[2]*p
  dot(Xp, v[3])-sum(log(1+exp(Xp)))
end

plogprior(p::Vector{Float64}, v::Vector) = -0.5*(dot(p, p)/v[1]+npars*log(2*pi*v[1]))

pgradlogtarget(p::Vector{Float64}, v::Vector) = v[2]'*(v[3]-1./(1+exp(-v[2]*p)))-p/v[1]

function ptensorlogtarget(p::Vector{Float64}, v::Vector)
  r = 1./(1+exp(-v[2]*p))
  broadcast(*, r.*(1-r), v[2])'*v[2]+(eye(npars)/v[1])
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

sampler = AM(0.02, 4)

mcrange = BasicMCRange(nsteps=nmcmc, burnin=nburnin)

outopts = Dict{Symbol, Any}(:monitor=>[:value], :diagnostics=>[:accept])

v0 = Dict(:λ=>100., :X=>covariates, :y=>outcome, :p=>rand(Normal(0, 3), npars))

job = BasicMCJob(model, sampler, mcrange, v0, tuner=VanillaMCTuner(verbose=false), outopts=outopts)

tic()
run(job)
runtime = toc()

chain = output(job)

acceptance(chain)

mean(chain)

ess(chain)
