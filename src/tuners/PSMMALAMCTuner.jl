type PSMMALAMCTune <: MCTunerState
  smmalatune::BasicMCTune
  malatune::BasicMCTune
  totaltune::BasicMCTune
end

PSMMALAMCTune(smmalastep::Real=1., malastep::Real=1.) =
  PSMMALAMCTune(BasicMCTune(smmalastep), BasicMCTune(malastep), BasicMCTune(1.))

immutable PSMMALAMCTuner <: MCTuner
  smmalatuner::Union{VanillaMCTuner, AcceptanceRateMCTuner}
  malatuner::Union{VanillaMCTuner, AcceptanceRateMCTuner}
  totaltuner::VanillaMCTuner
  smmalathreshold::Real
  malathreshold::Real
end

function tune!(tune::PSMMALAMCTune, tuner::PSMMALAMCTuner, ::Type{Val{:smmala}})
  if tune.smmalatune.proposed/tune.totaltune.proposed >= tuner.smmalathreshold
    tune.smmalatune.step *= tuner.smmalatuner.score(tune.smmalatune.rate-tuner.smmalatuner.targetrate)
  end
end

function tune!(tune::PSMMALAMCTune, tuner::PSMMALAMCTuner, ::Type{Val{:mala}})
  if tune.malatune.proposed/tune.totaltune.proposed >= tuner.malathreshold
    tune.malatune.step *= tuner.malatuner.score(tune.malatune.rate-tuner.malatuner.targetrate)
  end
end

Base.show(io::IO, tuner::PSMMALAMCTuner) = print(io, "PSMMALAMCTuner")

Base.writemime(io::IO, ::MIME"text/plain", tuner::PSMMALAMCTuner) = show(io, tuner)
