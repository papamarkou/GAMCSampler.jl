library(data.table)
library(stringr)

SAMPLERDIRS <- c("mala", "smmala_reverse", "asmmala")

DATADIR <- "../../data"
OUTDIR <- "../output"

nsamplerdirs <- length(SAMPLERDIRS)

nchains <- 10
nmcmc <- 110000
nburnin <- 10000
npostburnin <- nmcmc-nburnin

maxlag <- 50
ci <- 7
pi <- 18

cors <- matrix(data=NA, nrow=maxlag+1, ncol=nsamplerdirs)

for (j in 1:nsamplerdirs) {
  chains <- t(fread(
    file.path(DATADIR, SAMPLERDIRS[j], paste("chain", str_pad(ci, 2, pad="0"), ".csv", sep="")), sep=",", header=FALSE
  ))

  cors[, j] <- acf(chains[, pi], lag.max=maxlag, demean=TRUE, plot=FALSE)$acf
}

sqrtnpostburnin <- sqrt(npostburnin)

cols <- c("green", "blue", "red")

pdf(file=file.path(OUTDIR, "tdist_acfplot.pdf"), width=10, height=6)

plot(
  0:maxlag,
  cors[, 1],
  type="o",
  ylim=c(-0.2, 1.),
  col=cols[1],
  lwd=2,
  pch=20,
  xlab="",
  ylab="",
  cex.axis=1.8,
  cex.lab=1.7,
  yaxt="n"
)

axis(
  2,
  at=seq(-0.2, 1, by=0.2),
  labels=seq(-0.2, 1, by=0.2),
  cex.axis=1.8,
  las=1
)

lines(
  0:maxlag,
  cors[, 2],
  type="o",
  col=cols[2],
  lwd=2,
  pch=20
)

lines(
  0:maxlag,
  cors[, 3],
  type="o",
  col=cols[3],
  lwd=2,
  pch=20
)

legend(
  36, 0.6,
  c("MALA", "SMMALA", "ASMMALA"),
  lty=c(1, 1, 1),
  lwd=c(5, 5, 5),
  col=cols,
  cex=1.5,
  bty="n"
)

sqrt(npostburnin)

abline(h=1.96/sqrtnpostburnin, lty=2)
abline(h=-1.96/sqrtnpostburnin, lty=2)

dev.off()
