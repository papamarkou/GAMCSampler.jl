CURRENTDIR, CURRENTFILE = splitdir(@__FILE__)
ROOTDIR = splitdir(CURRENTDIR)[1]
OUTDIR = joinpath(ROOTDIR, "output")

# OUTDIR = "../output"

samplers = ["MALA", "AM", "SMMALA", "MAMALA"]

base_speed = readcsv(joinpath(OUTDIR, "MALA", "summary.csv"))[end]

open(joinpath(OUTDIR, "summary.txt"), "w") do f
  for s in samplers
    out = readcsv(joinpath(OUTDIR, s, "summary.csv"))
    write(
      f,
      @sprintf(
        "%s & %.2f & %d & %d & %d & %d & %.2f & %.2f & %.2f\\\\\n",
        s,
        out[1],
        minimum(out[2:21]),
        mean(out[2:21]),
        median(out[2:21]),
        maximum(out[2:21]),
        out[22],
        out[end],
        out[end]/base_speed
      )
    )
  end
end
