type PSMMALAMCTune <: MCTunerState
  smmalatune::BasicMCTune
  malatune::BasicMCTune
  totaltune::BasicMCTune
end

PSMMALAMCTune(smmalastep::Real=1., malastep::Real=1.) =
  PSMMALAMCTune(BasicMCTune(smmalastep), BasicMCTune(malastep), BasicMCTune(1.))
