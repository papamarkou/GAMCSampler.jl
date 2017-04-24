#!/bin/bash

JULIABIN=julia

samplers=(AM MALA SMMALA MAMALA)

for sampler in ${samplers[@]}
do
  echo "Running $sampler..."
  $JULIABIN "$HOME/.julia/v0.5/MAMALASampler/examples/radial_velocity/src/two_planets/$sampler/simulations.jl"
done
