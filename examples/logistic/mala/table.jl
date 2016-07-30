using Lora

DATADIR = "data"
OUTDIR = "output"

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

  println("Iteration ", i, " of ", nchains, " completed")
end

results[:rate] = mean(ratio)
results[:ess] = mean(essizes, 1)
results[:time] = mean(readdlm(joinpath(DATADIR, "times.csv"), ',', Float64))
results[:efficiency] = minimum(results[:ess])/results[:time]

writedlm(
  joinpath(OUTDIR, "malasummary.txt"),
  round(Float64[results[:rate] results[:ess] results[:time] results[:efficiency]], 2),
  " & "
)
