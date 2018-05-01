#!/bin/bash

JULIABIN=julia

samplers=(AM MALA SMMALA GAMC)

for sampler in ${samplers[@]}
do
  echo "Running $sampler..."
  $JULIABIN "$HOME/.julia/v0.6/GAMCSampler/examples/radial_velocity/src/one_planet/$sampler/simulations.jl"
done
