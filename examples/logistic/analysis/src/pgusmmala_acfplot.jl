using Gadfly
using Lora
using StatsBase

DATADIR = "../../data"
SUBDATADIR = "pgusmmala"
OUTDIR = "../output"

npars = 4

nmcmc = 110000
nburnin = 10000
npostburnin = nmcmc-nburnin

nmeans = 10000
ci = 6
pi = 2

lags = 0:50

chains = readdlm(joinpath(DATADIR, SUBDATADIR, "chain"*lpad(string(ci), 2, 0)*".csv"), ',', Float64)

cors = autocor(vec(chains[pi, :]), lags, demean=true)

acfplot = plot(
  x=collect(lags),
  y=cors,
  Geom.line,
  Geom.point,
  Coord.Cartesian(ymin=0., ymax=1.),
  Guide.xlabel(""),
  Guide.ylabel(""),
  Guide.title("")
)

draw(PDF(joinpath(OUTDIR, "logit_pgusmmala_acfplot.pdf"), 10cm, 7cm), acfplot)
