#!/bin/bash

RBIN=Rscript

$RBIN "$HOME/.julia/v0.5/MAMALASampler/examples/t_distribution/src/acfplot.r"
$RBIN "$HOME/.julia/v0.5/MAMALASampler/examples/t_distribution/src/meanplot.r"
$HOME/.julia/v0.5/MAMALASampler/examples/t_distribution/src/traceplots.sh
