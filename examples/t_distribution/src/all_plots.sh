#!/bin/bash

RBIN=Rscript

$RBIN "$HOME/.julia/v0.6/GAMCSampler/examples/t_distribution/src/acfplot.r"
$RBIN "$HOME/.julia/v0.6/GAMCSampler/examples/t_distribution/src/meanplot.r"
$HOME/.julia/v0.6/GAMCSampler/examples/t_distribution/src/traceplots.sh
