using Distributions
using Klara
using MAMALASampler


SRCDIR = "../../../src"
DATADIR = "../../../data"
OUTDIR = "../../../output/one_planet"

SUBOUTDIR = "MALA"

include(joinpath(SRCDIR, "rv_model.jl"))
include(joinpath(SRCDIR, "utils_ex.jl"))

using RvModelKeplerian

nchains = 1
nmcmc = 110000
nburnin = 10000

dataset = readdlm(joinpath(DATADIR, "one_planet.csv"), ',', header=false); # read observational data
obs_times = dataset[:, 1]
obs_rv = dataset[:, 2]
sigma_obs = dataset[:, 3]
set_times(obs_times); # set data to use for model evaluation
set_obs( obs_rv);
set_sigma_obs(sigma_obs);

param_true = make_param_true_ex1()
param_perturb_scale = make_param_perturb_scale(param_true)

p = BasicContMuvParameter(:p, logtarget=plogtarget, diffopts=DiffOptions(mode=:forward))

model = likelihood_model(p, false)

sampler = MALA(0.02)

mcrange = BasicMCRange(nsteps=nmcmc, burnin=nburnin)

mctuner = AcceptanceRateMCTuner(0.574, score=x -> logistic_rate_score(x, 3.), verbose=false)

outopts = Dict{Symbol, Any}(:monitor=>[:value], :diagnostics=>[:accept])

chain = nothing
times = Array(Float64, nchains)
stepsizes = Array(Float64, nchains)
i = 1

while i <= nchains
  param_init = param_true.+0.01*param_perturb_scale.*randn(length(param_true))
  v0 = Dict(:p=>param_init)

  job = BasicMCJob(model, sampler, mcrange, v0, tuner=mctuner, outopts=outopts)

  tic()
  run(job)
  runtime = toc()

  chain = output(job)
  ratio = acceptance(chain)

  if 0.05 < ratio < 0.95
    times[i] = runtime
    stepsizes[i] = job.sstate.tune.step

    println("Iteration ", i, " of ", nchains, " completed with acceptance ratio ", ratio)
    i += 1
  end
end

mean(chain)
param_true-mean(chain)

acceptance(chain)

ess(chain)
