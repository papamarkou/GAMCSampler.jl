#!/bin/bash

RBIN=Rscript

$RBIN "$HOME/.julia/v0.5/MAMALASampler/examples/radial_velocity/src/two_planets/acfplot.r"
$RBIN "$HOME/.julia/v0.5/MAMALASampler/examples/radial_velocity/src/two_planets/meanplot.r"
$HOME/.julia/v0.5/MAMALASampler/examples/radial_velocity/src/two_planets/traceplots.sh
