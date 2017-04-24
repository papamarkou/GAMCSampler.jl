CURRENTDIR, CURRENTFILE = splitdir(@__FILE__)
ROOTDIR = splitdir(splitdir(CURRENTDIR)[1])[1]
OUTDIR = joinpath(ROOTDIR, "output", "two_planets")

# OUTDIR = "../../output/two_planets"

samplers = ["MALA", "AM", "SMMALA", "MAMALA"]

base_speed = readcsv(joinpath(OUTDIR, "MALA", "summary.csv"))[end]

open(joinpath(OUTDIR, "summary.txt"), "w") do f
  for s in samplers
    out = readcsv(joinpath(OUTDIR, s, "summary.csv"))
    write(
      f,
      @sprintf("%s & %.2f & %d & %d & %d & %d & %.2f & %.2f & %.2f\\\\\n", s, out..., out[end]/base_speed)
    )
  end
end
