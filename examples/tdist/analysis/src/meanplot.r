library(data.table)
library(stringr)

SAMPLERDIRS <- c("mala", "smmala_reverse", "alsmmala", "amsmmala")

DATADIR <- "../../data"
OUTDIR <- "../output"

nsamplerdirs <- length(SAMPLERDIRS)

nchains <- 10
nmcmc <- 110000
nburnin <- 10000
npostburnin <- nmcmc-nburnin

nmeans <- 50000
ci <- 3
pi <- 7

submeans <- matrix(data=NA, nrow=nmeans, ncol=nsamplerdirs)

for (j in 1:nsamplerdirs) {
  chains <- t(fread(
    file.path(DATADIR, SAMPLERDIRS[j], paste("chain", str_pad(ci, 2, pad="0"), ".csv", sep="")), sep=",", header=FALSE
  ))

  for (i in 1:nmeans) {
    submeans[i, j] <- mean(chains[1:i, pi])
  }
}

cols <- c("green", "blue", "orange", "red")

pdf(file=file.path(OUTDIR, "tdist_meanplot.pdf"), width=10, height=6)

plot(
  1:nmeans,
  submeans[, 1],
  type="l",
  ylim=c(-1., 1.5),
  col=cols[1],
  lwd=2,
  xlab="",
  ylab="",
  cex.axis=1.8,
  cex.lab=1.7,
  yaxt="n"
)

axis(
  2,
  at=c(-1, -0.5, 0, 0.5, 1, 1.5),
  labels=c(-1, -0.5, 0, 0.5, 1, 1.5),
  cex.axis=1.8,
  las=1
)

lines(
  1:nmeans,
  submeans[, 2],
  type="l",
  col=cols[2],
  lwd=2
)

lines(
  1:nmeans,
  submeans[, 3],
  type="l",
  col=cols[3],
  lwd=2
)

lines(
  1:nmeans,
  submeans[, 4],
  type="l",
  col=cols[4],
  lwd=2
)

legend(
  35000, 1.5,
  c("MALA", "SMMALA", "ALSMMALA", "AMSMMALA"),
  lty=c(1, 1, 1),
  lwd=c(5, 5, 5),
  col=cols,
  cex=1.5,
  bty="n",
  text.width=4000
)

dev.off()
