function iterate!(job::BasicMCJob, ::Type{GAMC}, ::Type{Multivariate})
  job.sstate.count += 1

  if (
    job.tuner.totaltuner.verbose ||
    job.tuner.smmalatuner.verbose ||
    job.tuner.amtuner.verbose ||
    isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
  )
    job.sstate.tune.totaltune.proposed += 1
  end

  if job.sstate.count > job.sampler.t0
    job.sampler.update!(job.sstate, job.pstate, job.sstate.count, job.range.nsteps)
  end

  if job.sstate.presentupdatetensor
    job.sstate.updatetensorcount += 1

    if job.tuner.smmalatuner.verbose
      job.sstate.tune.smmalatune.proposed += 1
    end

    if !_job.sstate.pastupdatetensor
      job.parameter.uptotensorlogtarget!(job.pstate)

      if job.sampler.transform != nothing
        job.pstate.tensorlogtarget[:, :] = job.sampler.transform(job.pstate.tensorlogtarget)
      end

      job.sstate.oldinvtensor[:, :] = inv(job.pstate.tensorlogtarget)

      job.sstate.oldfirstterm[:] = job.sstate.oldinvtensor*job.pstate.gradlogtarget

      job.sstate.cholinvtensor[:, :] = ctranspose(chol(Hermitian(job.sstate.oldinvtensor)))
    end

    job.sstate.μ[:] = job.pstate.value+0.5*job.sstate.tune.totaltune.step*job.sstate.oldfirstterm

    job.sstate.pstate.value[:] = job.sstate.μ+job.sstate.sqrttunestep*job.sstate.cholinvtensor*randn(job.pstate.size)

    job.parameter.uptotensorlogtarget!(job.sstate.pstate)

    if job.sampler.transform != nothing
      job.sstate.pstate.tensorlogtarget[:, :] = job.sampler.transform(job.sstate.pstate.tensorlogtarget)
    end

    job.sstate.newinvtensor[:, :] = inv(job.sstate.pstate.tensorlogtarget)

    job.sstate.ratio = job.sstate.pstate.logtarget-job.pstate.logtarget

    job.sstate.ratio += (
      0.5*(
        logdet(job.sstate.tune.totaltune.step*job.sstate.oldinvtensor)+dot(
          job.sstate.pstate.value-job.sstate.μ,
          job.pstate.tensorlogtarget*(job.sstate.pstate.value-job.sstate.μ)
        )/job.sstate.tune.totaltune.step
      )
    )

    job.sstate.newfirstterm[:] = job.sstate.newinvtensor*job.sstate.pstate.gradlogtarget

    job.sstate.μ[:] = job.sstate.pstate.value+0.5*job.sstate.tune.totaltune.step*job.sstate.newfirstterm

    job.sstate.ratio -= (
      0.5*(
        logdet(job.sstate.tune.totaltune.step*job.sstate.newinvtensor)+dot(
          job.pstate.value-job.sstate.μ,
          job.sstate.pstate.tensorlogtarget*(job.pstate.value-job.sstate.μ)
        )/_job.sstate.tune.totaltune.step
      )
    )

    if job.sstate.ratio > 0 || (job.sstate.ratio > log(rand()))
      job.pstate.value = copy(job.sstate.pstate.value)

      job.pstate.gradlogtarget = copy(job.sstate.pstate.gradlogtarget)

      if in(:gradloglikelihood, job.outopts[:monitor]) && job.parameter.gradloglikelihood! != nothing
        job.pstate.gradloglikelihood = copy(job.sstate.pstate.gradloglikelihood)
      end

      if in(:gradlogprior, job.outopts[:monitor]) && job.parameter.gradlogprior! != nothing
        job.pstate.gradlogprior = copy(job.sstate.pstate.gradlogprior)
      end

      job.pstate.tensorlogtarget = copy(job.sstate.pstate.tensorlogtarget)

      if in(:tensorloglikelihood, job.outopts[:monitor]) && job.parameter.tensorloglikelihood! != nothing
        job.pstate.tensorloglikelihood = copy(job.sstate.pstate.tensorloglikelihood)
      end

      if in(:tensorlogprior, job.outopts[:monitor]) && job.parameter.tensorlogprior! != nothing
        job.pstate.tensorlogprior = copy(job.sstate.pstate.tensorlogprior)
      end

      job.sstate.oldinvtensor = copy(job.sstate.newinvtensor)

      job.sstate.cholinvtensor[:, :] = ctranspose(chol(Hermitian(job.sstate.newinvtensor)))

      job.sstate.oldfirstterm = copy(job.sstate.newfirstterm)

      job.pstate.logtarget = job.sstate.pstate.logtarget

      if in(:loglikelihood, job.outopts[:monitor]) && job.parameter.loglikelihood! != nothing
        job.pstate.loglikelihood = job.sstate.pstate.loglikelihood
      end

      if in(:logprior, job.outopts[:monitor]) && job.parameter.logprior! != nothing
        job.pstate.logprior = job.sstate.pstate.logprior
      end

      if !isempty(job.sstate.diagnosticindices)
        if haskey(job.sstate.diagnosticindices, :accept)
          job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:accept]] = true
        end
      end

      if job.tuner.totaltuner.verbose || isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
        job.sstate.tune.totaltune.accepted += 1
      end

      if job.tuner.smmalatuner.verbose
        job.sstate.tune.smmalatune.accepted += 1
      end
    else
      if !isempty(job.sstate.diagnosticindices)
        if haskey(job.sstate.diagnosticindices, :accept)
          job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:accept]] = false
        end
      end
    end
  else
    if job.tuner.amtuner.verbose
      job.sstate.tune.amtune.proposed += 1
    end

    covariance!(
      job.sstate.oldinvtensor,
      job.sstate.oldinvtensor,
      job.sstate.count-2,
      job.pstate.value,
      job.sstate.lastmean,
      job.sstate.secondlastmean
    )

    job.sstate.oldinvtensor[:, :] = Hermitian(job.sstate.oldinvtensor)

    set_gmm!(job.sstate, job.sampler, job.pstate)

    job.sstate.pstate.value[:] =  rand(job.sstate.proposal)

    job.parameter.logtarget!(job.sstate.pstate)

    job.sstate.ratio = job.sstate.pstate.logtarget-job.pstate.logtarget

    job.sstate.ratio -= logpdf(job.sstate.proposal, job.sstate.pstate.value)

    set_gmm!(job.sstate, job.sampler, job.sstate.pstate)

    job.sstate.ratio += logpdf(job.sstate.proposal, job.pstate.value)

    if job.sstate.ratio > 0 || (job.sstate.ratio > log(rand()))
      job.pstate.value = copy(job.sstate.pstate.value)

      job.pstate.logtarget = job.sstate.pstate.logtarget

      if in(:loglikelihood, job.outopts[:monitor]) && job.parameter.loglikelihood! != nothing
        job.pstate.loglikelihood = job.sstate.pstate.loglikelihood
      end

      if in(:logprior, job.outopts[:monitor]) && job.parameter.logprior! != nothing
        job.pstate.logprior = job.sstate.pstate.logprior
      end

      if !isempty(job.sstate.diagnosticindices)
        if haskey(job.sstate.diagnosticindices, :accept)
          job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:accept]] = true
        end
      end

      if job.tuner.totaltuner.verbose || isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
        job.sstate.tune.totaltune.accepted += 1
      end

      if job.tuner.amtuner.verbose
        job.sstate.tune.amtune.accepted += 1
      end
    else
      if !isempty(job.sstate.diagnosticindices)
        if haskey(job.sstate.diagnosticindices, :accept)
          job.pstate.diagnosticvalues[job.sstate.diagnosticindices[:accept]] = false
        end
      end
    end
  end

  job.sstate.secondlastmean = copy(job.sstate.lastmean)

  recursive_mean!(job.sstate.lastmean, job.sstate.lastmean, job.sstate.count, job.pstate.value)

  job.sstate.pastupdatetensor = job.sstate.presentupdatetensor

  if (
    job.tuner.totaltuner.verbose ||
    job.tuner.smmalatuner.verbose ||
    job.tuner.amtuner.verbose ||
    isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
  )
    if (
      job.sstate.tune.totaltune.totproposed <= job.range.burnin &&
      mod(job.sstate.tune.totaltune.proposed, job.tuner.totaltuner.period) == 0
    )
      if job.tuner.totaltuner.verbose || isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
        rate!(job.sstate.tune.totaltune)
      end

      if job.tuner.smmalatuner.verbose
        job.sstate.tune.smmalafrequency = job.sstate.tune.smmalatune.proposed/job.sstate.tune.totaltune.proposed
        rate!(job.sstate.tune.smmalatune)
      end

      if job.tuner.amtuner.verbose
        job.sstate.tune.amfrequency = job.sstate.tune.amtune.proposed/job.sstate.tune.totaltune.proposed
        rate!(job.sstate.tune.amtune)
      end

      if job.tuner.totaltuner.verbose || job.tuner.smmalatuner.verbose || job.tuner.amtuner.verbose
        println("Burnin iteration ", job.fmt_iter(job.sstate.count), " out of ", job.range.burnin, "...")

        if job.tuner.totaltuner.verbose
          println(
            "  Total : ",
            job.sstate.fmt_iter(job.sstate.tune.totaltune.accepted),
            "/",
            job.sstate.fmt_iter(job.sstate.tune.totaltune.proposed),
            " (",
            job.fmt_perc(100*job.sstate.tune.totaltune.rate),
            "%) acceptance rate"
          )
        end

        if job.tuner.smmalatuner.verbose
          println(
            "  SMMALA: ",
            job.sstate.fmt_iter(job.sstate.tune.smmalatune.accepted),
            "/",
            job.sstate.fmt_iter(job.sstate.tune.smmalatune.proposed),
            " (",
            job.fmt_perc(100*job.sstate.tune.smmalatune.rate),
            "%) acceptance rate, ",
            job.sstate.fmt_iter(job.sstate.tune.smmalatune.proposed),
            "/",
            job.sstate.fmt_iter(job.sstate.tune.totaltune.proposed),
            " (",
            job.fmt_perc(100*job.sstate.tune.smmalafrequency),
            "%) running frequency"
          )
        end

        if job.tuner.amtuner.verbose
          println(
            "  AM    : ",
            job.sstate.fmt_iter(job.sstate.tune.amtune.accepted),
            "/",
            job.sstate.fmt_iter(job.sstate.tune.amtune.proposed),
            " (",
            job.fmt_perc(100*job.sstate.tune.amtune.rate),
            "%) acceptance rate, ",
            job.sstate.fmt_iter(job.sstate.tune.amtune.proposed),
            "/",
            job.sstate.fmt_iter(job.sstate.tune.totaltune.proposed),
            " (",
            job.fmt_perc(100*job.sstate.tune.amfrequency),
            "%) running frequency"
          )
        end
      end

      if isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
        tune!(job.sstate.tune.totaltune, job.tuner.totaltuner)
        job.sstate.sqrttunestep = sqrt(job.sstate.tune.totaltune.step)
      end

      if job.tuner.smmalatuner.verbose
        reset_burnin!(job.sstate.tune.smmalatune)
      end

      if job.tuner.amtuner.verbose
        reset_burnin!(job.sstate.tune.amtune)
      end

      job.sstate.tune.totaltune.totproposed += job.sstate.tune.totaltune.proposed

      job.sstate.tune.totaltune.proposed = 0

      if job.tuner.totaltuner.verbose || isa(job.tuner.totaltuner, AcceptanceRateMCTuner)
        job.sstate.tune.totaltune.accepted = 0
        job.sstate.tune.totaltune.rate = NaN
      end
    end
  end
end
