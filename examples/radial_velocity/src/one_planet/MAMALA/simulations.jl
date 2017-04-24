using Distributions
using Klara
using MAMALASampler

CURRENTDIR, CURRENTFILE = splitdir(@__FILE__)
ROOTDIR = splitdir(splitdir(splitdir(CURRENTDIR)[1])[1])[1]
SRCDIR = joinpath(ROOTDIR, "src")
DATADIR = joinpath(ROOTDIR, "data")
OUTDIR = joinpath(ROOTDIR, "output", "one_planet")

# SRCDIR = "../../../src"
# DATADIR = "../../../data"
# OUTDIR = "../../../output/one_planet"

SUBOUTDIR = "MAMALA"

include(joinpath(SRCDIR, "rv_model.jl"))
include(joinpath(SRCDIR, "utils_ex.jl"))

using RvModelKeplerian

nchains = 10
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

p = BasicContMuvParameter(:p, logtarget=plogtarget, diffopts=DiffOptions(mode=:forward, order=2))

model = likelihood_model(p, false)

sampler = MAMALA(
  update=(sstate, pstate, i, tot) -> rand_exp_decay_update!(sstate, pstate, i, tot, 10.),
  transform=H -> softabs(H, 1000.),
  driftstep=0.02,
  minorscale=0.001,
  c=0.01
)

mcrange = BasicMCRange(nsteps=nmcmc, burnin=nburnin)

mctuner = MAMALAMCTuner(
  VanillaMCTuner(verbose=false),
  VanillaMCTuner(verbose=false),
  AcceptanceRateMCTuner(0.35, verbose=false)
)

outopts = Dict{Symbol, Any}(:monitor=>[:value], :diagnostics=>[:accept])

times = Array(Float64, nchains)
stepsizes = Array(Float64, nchains)
nupdates = Array(Int64, nchains)
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

  if 0.22 < ratio < 0.37
    writedlm(joinpath(OUTDIR, SUBOUTDIR, "chain"*lpad(string(i), 2, 0)*".csv"), chain.value, ',')
    writedlm(joinpath(OUTDIR, SUBOUTDIR, "diagnostics"*lpad(string(i), 2, 0)*".csv"), vec(chain.diagnosticvalues), ',')

    times[i] = runtime
    stepsizes[i] = job.sstate.tune.totaltune.step
    nupdates[i] = job.sstate.updatetensorcount

    println("Iteration ", i, " of ", nchains, " completed with acceptance ratio ", ratio)
    i += 1
  end
end

writedlm(joinpath(OUTDIR, SUBOUTDIR, "times.csv"), times, ',')
writedlm(joinpath(OUTDIR, SUBOUTDIR, "stepsizes.csv"), stepsizes, ',')
writedlm(joinpath(OUTDIR, SUBOUTDIR, "nupdates.csv"), nupdates, ',')
