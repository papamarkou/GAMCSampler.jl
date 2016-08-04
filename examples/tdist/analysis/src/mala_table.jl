using Lora

DATADIR = "../../data"
SUBDATADIR = "mala"
OUTDIR = "../output"

npars = 20

nchains = 10
nmcmc = 110000
nburnin = 10000

ratio = Array(Float64, nchains)
essizes = Array(Float64, nchains, npars)
results = Dict{Symbol, Any}()

for i in 1:nchains
  ratio[i] = acceptance(readdlm(joinpath(DATADIR, SUBDATADIR, "diagnostics"*lpad(string(i), 2, 0)*".csv"), ',', Bool))
  essizes[i, :] = ess(readdlm(joinpath(DATADIR, SUBDATADIR, "chain"*lpad(string(i), 2, 0)*".csv"), ',', Float64), 2)

  println("Iteration ", i, " of ", nchains, " completed")
end

results[:rate] = mean(ratio)
results[:ess] = mean(essizes, 1)
results[:time] = mean(readdlm(joinpath(DATADIR, SUBDATADIR, "times.csv"), ',', Float64))
results[:efficiency] = minimum(results[:ess])/results[:time]

writedlm(
  joinpath(OUTDIR, "tdist_mala_summary.csv"),
  hcat(
    results[:rate],
    results[:ess],
    results[:time],
    results[:efficiency]
  ),
  ','
)

writedlm(
  joinpath(OUTDIR, "tdist_mala_summary.txt"),
  Any[
    round(results[:rate], 2)
    [Int64(i) for i in round(results[:ess])]
    round(results[:time], 2)
    round(results[:efficiency], 2)
  ]',
  " & "
)
