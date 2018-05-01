CURRENTDIR, CURRENTFILE = splitdir(@__FILE__)
ROOTDIR = splitdir(CURRENTDIR)[1]
OUTDIR = joinpath(ROOTDIR, "output")

# OUTDIR = "../output"

samplers = ["MALA", "AM", "SMMALA", "GAMC"]

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
