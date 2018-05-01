#!/bin/bash

RBIN=Rscript

$RBIN "$HOME/.julia/v0.6/GAMCSampler/examples/radial_velocity/src/one_planet/acfplot.r"
$RBIN "$HOME/.julia/v0.6/GAMCSampler/examples/radial_velocity/src/one_planet/meanplot.r"
$HOME/.julia/v0.6/GAMCSampler/examples/radial_velocity/src/one_planet/traceplots.sh
