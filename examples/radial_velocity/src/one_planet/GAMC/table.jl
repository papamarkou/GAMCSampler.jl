using Klara

CURRENTDIR, CURRENTFILE = splitdir(@__FILE__)
ROOTDIR = splitdir(splitdir(splitdir(CURRENTDIR)[1])[1])[1]
OUTDIR = joinpath(ROOTDIR, "output", "one_planet")

# OUTDIR = "../../output/one_planet"

SUBOUTDIR = "GAMC"

npars = 6

nchains = 10
nmcmc = 110000
nburnin = 10000

ratio = Array{Float64}(nchains)
essizes = Array{Float64}(nchains, npars)
results = Dict{Symbol, Any}()

for i in 1:nchains
  ratio[i] = acceptance(readdlm(joinpath(OUTDIR, SUBOUTDIR, "diagnostics"*lpad(string(i), 2, 0)*".csv"), ',', Bool))
  essizes[i, :] = ess(readdlm(joinpath(OUTDIR, SUBOUTDIR, "chain"*lpad(string(i), 2, 0)*".csv"), ',', Float64), 2)

  println("Iteration ", i, " of ", nchains, " completed")
end

results[:rate] = mean(ratio)
results[:ess] = mean(essizes, 1)
results[:time] = mean(readdlm(joinpath(OUTDIR, SUBOUTDIR, "times.csv"), ',', Float64))
results[:efficiency] = minimum(results[:ess])/results[:time]

writedlm(
  joinpath(OUTDIR, SUBOUTDIR, "summary.csv"),
  hcat(
    results[:rate],
    results[:ess],
    results[:time],
    results[:efficiency]
  ),
  ','
)

writedlm(
  joinpath(OUTDIR, SUBOUTDIR, "summary.txt"),
  Any[
    round(results[:rate], 2)
    [Int64(i) for i in round(results[:ess])]...
    round(results[:time], 2)
    round(results[:efficiency], 2)
  ]',
  " & "
)
