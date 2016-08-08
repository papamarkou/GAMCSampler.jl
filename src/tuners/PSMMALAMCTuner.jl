type PSMMALAMCTune <: MCTunerState
  smmalatune::BasicMCTune
  malatune::BasicMCTune
  totaltune::BasicMCTune
end

PSMMALAMCTune(smmalastep::Real=1., malastep::Real=1.) =
  PSMMALAMCTune(BasicMCTune(smmalastep), BasicMCTune(malastep), BasicMCTune(1.))

accepted!(tune::PSMMALAMCTune) = tune.totaltune.accepted = tune.smmalatune.accepted+tune.malatune.accepted

proposed!(tune::PSMMALAMCTune) = tune.totaltune.proposed = tune.smmalatune.proposed+tune.malatune.proposed

totproposed!(tune::PSMMALAMCTune) = tune.totaltune.totproposed = tune.smmalatune.totproposed+tune.malatune.totproposed
