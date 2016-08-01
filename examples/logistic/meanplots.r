library(data.table)
library(stringr)

SAMPLERDIRS <- c("mala", "smmala", "pgusmmala")

DATADIR <- "data"
OUTDIR <- "output"

nsamplerdirs <- length(SAMPLERDIRS)

npars <- 4

nchains <- 10
nmcmc <- 110000
nburnin <- 10000
npostburnin <- nmcmc-nburnin

nmeans <- 10000
ci <- 6
pi <- 2

submeans <- matrix(data=NA, nrow=nmeans, ncol=nsamplerdirs)

for (j in 1:nsamplerdirs) {
  chains <- t(fread(
    file.path(SAMPLERDIRS[j], DATADIR, paste("chain", str_pad(ci, 2, pad="0"), ".csv", sep="")), sep=",", header=FALSE
  ))

  for (i in 1:nmeans) {
    submeans[i, j] <- mean(chains[1:i, pi])
  }
}

cols <- c("green", "blue", "red")

# pdf(file=file.path(OUTDIR, "logistic_traceplots.pdf"), width=10, height=6)

plot(
  1:nmeans,
  submeans[, 1],
  type="l",
  ylim=c(0.4, 1.2),
  col=cols[1],
  lwd=2,
  xlab="",
  ylab="",
  cex.axis=1.8,
  cex.lab=1.7
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

legend(
  "topright",
  c("MALA", "SMMALA", "PSMMALA"),
  lty=c(1, 1, 1),
  lwd=c(5, 5, 5),
  col=cols,
  cex=1.7
)

# dev.off()