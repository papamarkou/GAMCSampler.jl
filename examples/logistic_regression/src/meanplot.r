# library(data.table)
# library(stringr)

SAMPLERDIRS <- c("AM", "MALA", "SMMALA", "MAMALA")

OUTDIR <- "../output"

nsamplerdirs <- length(SAMPLERDIRS)

npars <- 4

nchains <- 10
nmcmc <- 110000
nburnin <- 10000
npostburnin <- nmcmc-nburnin

nmeans <- 50000
ci <- 9
pi <- 4

submeans <- matrix(data=NA, nrow=nmeans, ncol=nsamplerdirs)

for (j in 1:nsamplerdirs) {
  chains <- t(fread(
    file.path(OUTDIR, SAMPLERDIRS[j], paste("chain", str_pad(ci, 2, pad="0"), ".csv", sep="")), sep=",", header=FALSE
  ))

  for (i in 1:nmeans) {
    submeans[i, j] <- mean(chains[1:i, pi])
  }
}

cols <- c("green", "blue", "red", "orange")

# pdf(file=file.path(OUTDIR, "logit_meanplot.pdf"), width=10, height=6)

plot(
  1:nmeans,
  submeans[, 1],
  type="l",
  # ylim=c(0.5, 1),
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
  at=seq(0.5, 1, by=0.1),
  labels=seq(0.5, 1, by=0.1),
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
  35000, 1,
  c("MALA", "SMMALA", "AMSMMALA", "ALSMMALA"),
  lty=c(1, 1, 1),
  lwd=c(5, 5, 5),
  col=cols,
  cex=1.5,
  bty="n",
  text.width=2000
)

# dev.off()
