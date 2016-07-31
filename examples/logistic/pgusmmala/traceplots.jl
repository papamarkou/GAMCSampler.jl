using Gadfly
using Lora

DATADIR = "data"
OUTDIR = "output"

npars = 4

nchains = 10
nmcmc = 110000
nburnin = 10000
npostburnin = nmcmc-nburnin

ci = 5
pi = 2

chains = readdlm(joinpath(DATADIR, "chain"*lpad(string(ci), 2, 0)*".csv"), ',', Float64)

chainmean = mean(chains[pi, :])

traceplot = plot(
  layer(
    x -> chainmean,
    0,
    npostburnin,
    Geom.line,
    Theme(default_color=colorant"red")
  ),
  layer(
    x=collect(1:npostburnin),
    y=chains[pi, :],
    Geom.line
  ),
  Coord.Cartesian(ymin=-2., ymax=3.),
  Guide.xticks(ticks=[0, npostburnin/2, npostburnin]),
  Theme(
    major_label_font_size=14pt,
    minor_label_font_size=12pt,
    grid_line_width=0pt,
    panel_stroke=colorant"grey"
    ),
  Guide.xlabel(""),
  Guide.ylabel(""),
  Guide.title("")
)

draw(PDF(joinpath(OUTDIR, "logit_pgusmmala_traceplot.pdf"), 14cm, 7cm), traceplot)
