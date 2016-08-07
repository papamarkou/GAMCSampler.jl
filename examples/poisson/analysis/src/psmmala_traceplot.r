library(data.table)
library(stringr)

DATADIR <- "../../data"
SUBDATADIR <- "psmmala"
OUTDIR <- "../output"

npars <- 4

nmcmc <- 110000
nburnin <- 10000
npostburnin <- nmcmc-nburnin

nmeans <- 10000
ci <- 10
pi <- 2

chains <- t(fread(
  file.path(DATADIR, SUBDATADIR, paste("chain", str_pad(ci, 2, pad="0"), ".csv", sep="")), sep=",", header=FALSE
))

chainmean = mean(chains[, pi])

pdf(file=file.path(OUTDIR, "poisson_psmmala_traceplot.pdf"), width=10, height=6)

plot(
  1:npostburnin,
  chains[, pi],
  type="l",
  ylim=c(-0.02, 0.22),
  col="steelblue2",
  xlab="",
  ylab="",
  cex.axis=1.8,
  cex.lab=1.7,
  yaxt="n"
)

axis(
  2,
  at=seq(0, 0.2, by=0.1),
  labels=seq(0, 0.2, by=0.1),
  cex.axis=1.8,
  las=1
)

lines(
  1:npostburnin,
  rep(chainmean, npostburnin),
  type="l",
  col="orangered1",
  lwd=2
)

dev.off()
