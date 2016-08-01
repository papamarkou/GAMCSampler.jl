library(data.table)
library(stringr)

DATADIR <- "../../data"
SUBDATADIR <- "pgusmmala"
OUTDIR <- "../output"

npars <- 4

nmcmc <- 110000
nburnin <- 10000
npostburnin <- nmcmc-nburnin

nmeans <- 10000
ci <- 4
pi <- 2

chains <- t(fread(
  file.path(DATADIR, SUBDATADIR, paste("chain", str_pad(ci, 2, pad="0"), ".csv", sep="")), sep=",", header=FALSE
))

chainmean = mean(chains[, pi])

pdf(file=file.path(OUTDIR, "logit_pgusmmala_traceplot.pdf"), width=10, height=6)

plot(
  1:npostburnin,
  chains[, pi],
  type="l",
  ylim=c(-1.2, 3),
  col="cyan4",
  xlab="",
  ylab="",
  cex.axis=1.8,
  cex.lab=1.7,
  yaxt="n"
)

axis(
  2,
  at=seq(-1, 3, by=1),
  labels=seq(-1, 3, by=1),
  cex.axis=1.8,
  las=1
)

lines(
  1:npostburnin,
  rep(chainmean, npostburnin),
  type="l",
  col="red",
  lwd=2
)

dev.off()
