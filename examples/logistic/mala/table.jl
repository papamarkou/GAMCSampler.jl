using Lora

DATADIR = "data"

npars = 4

nchains = 10
nmcmc = 110000
nburnin = 10000

ratio = Array(Float64, nchains)
essizes = Array(Float64, nchains, npars)
results = Dict{Symbol, Any}()

for i in 1:nchains
  ratio[i] = acceptance(readdlm(joinpath(DATADIR, "diagnostics"*lpad(string(i), 2, 0)*".csv"), ',', Bool))

  essizes[i, :] = ess(readdlm(joinpath(DATADIR, "chain"*lpad(string(i), 2, 0)*".csv"), ',', Float64), 2)
end

results[:rate] = mean(ratio)

results[:time] = mean(readdlm(joinpath(DATADIR, "times.csv"), ',', Float64))
