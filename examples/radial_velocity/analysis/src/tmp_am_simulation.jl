using Distributions
using Klara
using MAMALASampler

include("rv_model.jl")

using RvModelKeplerian

DATADIR = "../../data"

dataset = readdlm(joinpath(DATADIR, "example1.csv"), ',', header=false);  # read observational data
obs_times = dataset[:,1]
obs_rv = dataset[:,2]
sigma_obs = dataset[:,3]
set_times(obs_times);     # set data to use for model evaluation
set_obs( obs_rv);
set_sigma_obs(sigma_obs);

include("utils_ex.jl")
param_true = make_param_true_ex1()
param_perturb_scale = make_param_perturb_scale(param_true)
param_init = 0
param_init = param_true.+0.01*param_perturb_scale.*randn(length(param_true))
println("param_init= ",param_init)

sampler = AM(0.01, 6)

nmcmc = 50000
nburnin = 10000
mcrange = BasicMCRange(nsteps=nmcmc, burnin=nburnin)

outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget], :diagnostics=>[:accept])

v0 = Dict(:p=>param_init)

p = BasicContMuvParameter(:p, logtarget=plogtarget)

model = likelihood_model(p, false)
#model = likelihood_model([p], isindexed=false)

target_accept_rate = 0.234

job = BasicMCJob(model, sampler, mcrange, v0, tuner=VanillaMCTuner(verbose=false), outopts=outopts)

tic()
run(job)
runtime = toc()

chain = output(job)

ratio = acceptance(chain)

mean(chain)
mean(chain)-param_true

ess(chain)
