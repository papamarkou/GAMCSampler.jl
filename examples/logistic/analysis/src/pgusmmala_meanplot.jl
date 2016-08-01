using Gadfly
using Lora

DATADIR = "../../data"
SUBDATADIR = "pgusmmala"
OUTDIR = "../output"

npars = 4

nchains = 10
nmcmc = 110000
nburnin = 10000
npostburnin = nmcmc-nburnin

nmeans = 10000
ci = 6
pi = 2

chains = readdlm(joinpath(DATADIR, SUBDATADIR, "chain"*lpad(string(ci), 2, 0)*".csv"), ',', Float64)

chainmean = mean(chains[pi, :])

submeans = Float64[mean(chains[pi, 1:i]) for i in 1:nmeans]

traceplot = plot(
  # layer(
  #   x -> chainmean,
  #   0,
  #   nmeans,
  #   Geom.line,
  #   Theme(default_color=colorant"red")
  # ),
  layer(
    x=collect(1:nmeans),
    y=submeans,
    Geom.line
  ),
  Coord.Cartesian(ymin=0.4, ymax=1.2),
  Guide.xticks(ticks=[0, nmeans/2, nmeans]),
  Guide.yticks(ticks=[0.4, 0.6, 0.8, 1.0, 1.2]),
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

draw(PDF(joinpath(OUTDIR, "logit_pgusmmala_meanplot.pdf"), 14cm, 7cm), traceplot)
