type PSMMALAMCTune <: MCTunerState
  smmalatune::BasicMCTune
  malatune::BasicMCTune
  totaltune::BasicMCTune
end

PSMMALAMCTune(smmalastep::Real=1., malastep::Real=1.) =
  PSMMALAMCTune(BasicMCTune(smmalastep), BasicMCTune(malastep), BasicMCTune(1.))

accepted!(tune::PSMMALAMCTune) = tune.totaltune.accepted = tune.smmalatune.accepted+tune.malatune.accepted

proposed!(tune::PSMMALAMCTune) = tune.totaltune.proposed = tune.smmalatune.proposed+tune.malatune.proposed

totrate!(tune::PSMMALAMCTune) = (tune.totaltune.rate = tune.totaltune.accepted/tune.totaltune.proposed)

function reset_totburnin!(tune::PSMMALAMCTune)
  tune.totaltune.totproposed += (tune.smmalatune.totproposed+tune.malatune.totproposed)
  (tune.accepted, tune.proposed, tune.rate) = (0, 0, NaN)
end

immutable PSMMALAMCTuner <: MCTuner
  smmalatuner::Union{VanillaMCTuner, AcceptanceRateMCTuner}
  malatuner::Union{VanillaMCTuner, AcceptanceRateMCTuner}
  totaltuner::VanillaMCTuner
end

Base.show(io::IO, tuner::PSMMALAMCTuner) = print(io, "PSMMALAMCTuner")

Base.writemime(io::IO, ::MIME"text/plain", tuner::PSMMALAMCTuner) = show(io, tuner)
