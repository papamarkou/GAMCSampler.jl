type PSMMALAMCTune <: MCTunerState
  smmalatune::BasicMCTune
  malatune::BasicMCTune
  totaltune::BasicMCTune
  smmalafrequency::Real
  malafrequency::Real
end

PSMMALAMCTune(smmalastep::Real=1., malastep::Real=1.) =
  PSMMALAMCTune(BasicMCTune(smmalastep), BasicMCTune(malastep), BasicMCTune(1.), NaN, NaN)

immutable PSMMALAMCTuner <: MCTuner
  smmalatuner::Union{VanillaMCTuner, AcceptanceRateMCTuner}
  malatuner::Union{VanillaMCTuner, AcceptanceRateMCTuner}
  totaltuner::VanillaMCTuner
  smmalathreshold::Real
  malathreshold::Real
end

function tune!(tune::PSMMALAMCTune, tuner::PSMMALAMCTuner, ::Type{Val{:smmala}})
  if tune.smmalafrequency >= tuner.smmalathreshold
    tune.smmalatune.step *= tuner.smmalatuner.score(tune.smmalatune.rate-tuner.smmalatuner.targetrate)
  end
end

function tune!(tune::PSMMALAMCTune, tuner::PSMMALAMCTuner, ::Type{Val{:mala}})
  if tune.malafrequency >= tuner.malathreshold
    tune.malatune.step *= tuner.malatuner.score(tune.malatune.rate-tuner.malatuner.targetrate)
  end
end

Base.show(io::IO, tuner::PSMMALAMCTuner) = print(io, "PSMMALAMCTuner")

Base.writemime(io::IO, ::MIME"text/plain", tuner::PSMMALAMCTuner) = show(io, tuner)
