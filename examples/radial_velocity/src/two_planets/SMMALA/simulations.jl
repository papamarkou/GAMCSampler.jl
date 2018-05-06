using Distributions
using Klara
using GAMCSampler

CURRENTDIR, CURRENTFILE = splitdir(@__FILE__)
ROOTDIR = splitdir(splitdir(splitdir(CURRENTDIR)[1])[1])[1]
SRCDIR = joinpath(ROOTDIR, "src")
DATADIR = joinpath(ROOTDIR, "data")
OUTDIR = joinpath(ROOTDIR, "output", "two_planets")

# SRCDIR = "../../src"
# DATADIR = "../../data"
# OUTDIR = "../../output/two_planets"

SUBOUTDIR = "SMMALA"

include(joinpath(SRCDIR, "rv_model.jl"))
include(joinpath(SRCDIR, "utils_ex.jl"))

using RvModelKeplerian

nchains = 10
nmcmc = 110000
nburnin = 10000

dataset = readdlm(joinpath(DATADIR, "two_planets.csv"), ',', header=false); # read observational data
obs_times = dataset[:, 1]
obs_rv = dataset[:, 2]
sigma_obs = dataset[:, 3]
set_times(obs_times); # set data to use for model evaluation
set_obs( obs_rv);
set_sigma_obs(sigma_obs);

param_true = make_param_true_ex2()
param_perturb_scale = make_param_perturb_scale(param_true)

p = BasicContMuvParameter(:p, logtarget=plogtarget, diffopts=DiffOptions(mode=:forward, order=2))

model = likelihood_model(p, false)

sampler = SMMALA(0.02, H -> softabs(H, 1000.))

mcrange = BasicMCRange(nsteps=nmcmc, burnin=nburnin)

mctuner = AcceptanceRateMCTuner(0.7, score=x -> logistic_rate_score(x, 3.), verbose=false)

outopts = Dict{Symbol, Any}(:monitor=>[:value], :diagnostics=>[:accept])

times = Array{Float64}(nchains)
stepsizes = Array{Float64}(nchains)
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

  if 0.45 < ratio < 0.85
    writedlm(joinpath(OUTDIR, SUBOUTDIR, "chain"*lpad(string(i), 2, 0)*".csv"), chain.value, ',')
    writedlm(joinpath(OUTDIR, SUBOUTDIR, "diagnostics"*lpad(string(i), 2, 0)*".csv"), vec(chain.diagnosticvalues), ',')

    times[i] = runtime
    stepsizes[i] = job.sstate.tune.step

    println("Iteration ", i, " of ", nchains, " completed with acceptance ratio ", ratio)
    i += 1
  end
end

writedlm(joinpath(OUTDIR, SUBOUTDIR, "times.csv"), times, ',')
writedlm(joinpath(OUTDIR, SUBOUTDIR, "stepsizes.csv"), stepsizes, ',')
