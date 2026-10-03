#pragma once
#include "algorithm_library/activity_detection.h"
#include "framework/framework.h"
#include "utilities/fastonebigheader.h"

// Estimate per-band signal activity from the ratio of input power to tracked noise
// and slow noise adaptation while a signal is active.
class ActivityDetectionNoiseEstimation : public AlgorithmImplementation<ActivityDetectionConfiguration, ActivityDetectionNoiseEstimation>
{
  public:
    ActivityDetectionNoiseEstimation(Coefficients c = Coefficients()) : BaseAlgorithm{c}
    {
        activityMean.resize(c.nBands, c.nChannels);
        powerNoise.resize(c.nBands, c.nChannels);
        activityMeanLambda = 1.f - expf(-1.f / (c.filterbankRate * .152f));
        resetVariables();
        onParametersChanged();
    }

    // return by const reference to avoid memory copy
    inline const Eigen::ArrayXXf &getPowerNoise() const { return powerNoise; }

  private:
    void processAlgorithm(Input powerNoisy, Output activity)
    {
        // activity detection
        activity = (powerNoisy / (powerNoise + 1e-20f) - 3.5f).cwiseMin(25.f);

        // for-loop has been profiled to be faster in gcc than the commented line below:
        // activity =activity.unaryExpr(std::ref(fasterExp));
        float *ptr = activity.data();
        for (auto i = 0; i < activity.size(); i++, ptr++)
        {
            *ptr = fasterexp(*ptr);
        }

        activity /= (1.f + activity);

        // smooth activity
        activityMean += activityMeanLambda * (activity - activityMean);
        activity = (activityMean > 0.99f).select(activity.cwiseMin(.99f), activity);

        // update noise
        powerNoise += smoothingLambda * (1.f - activity) * (powerNoisy - powerNoise);
    }

    void resetVariables() final
    {
        activityMean.setZero();
        powerNoise.setConstant(1e-12f);
    }

    void onParametersChanged() { smoothingLambda = 1.f - expf(-1.f / (C.filterbankRate * P.smoothingTConstant)); }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = activityMean.getDynamicMemorySize();
        size += powerNoise.getDynamicMemorySize();
        return size;
    }

    Eigen::ArrayXXf activityMean;
    Eigen::ArrayXXf powerNoise;
    float activityMeanLambda, smoothingLambda;

    friend BaseAlgorithm;
};

// Reduce per-band activity probabilities and channel-to-noise power ratios
// to one activity flag while retaining the underlying noise estimate.
class ActivityDetectionFusedNoiseEstimation : public AlgorithmImplementation<ActivityDetectionFusedConfiguration, ActivityDetectionFusedNoiseEstimation>
{
  public:
    ActivityDetectionFusedNoiseEstimation(Coefficients c = Coefficients()) : BaseAlgorithm{c}, activityDetection(convertToActivityDetectionConfiguration(c))
    {
        activity.resize(C.nBands, C.nChannels);
    }

    ActivityDetectionNoiseEstimation activityDetection;
    DEFINE_MEMBER_ALGORITHMS(activityDetection)

    // return by const reference to avoid memory copy
    inline const Eigen::ArrayXXf &getPowerNoise() const { return activityDetection.getPowerNoise(); }

  private:
    inline void processAlgorithm(Input powerNoisy, Output activityFlag)
    {
        activityDetection.process(powerNoisy, activity);
        activityFlag = (powerNoisy.colwise().mean() > (getPowerNoise().colwise().mean() * 1.5 * P.activityThreshold))
                           .any(); // mean power of each channel is compared. ActivityFlag is true if any channel is above threshold
        activityFlag |=
            static_cast<float>((activity > 0.5f * P.activityThreshold).rowwise().any().count()) >
            (0.15f * P.activityThreshold * C.nBands); // (1) true if activity is above threshold in any channel, (2) then count over bands and compare with threshold
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = activity.getDynamicMemorySize();
        return size;
    }

    ActivityDetectionNoiseEstimation::Coefficients convertToActivityDetectionConfiguration(const Coefficients &c)
    {
        ActivityDetectionNoiseEstimation::Coefficients c2;
        c2.filterbankRate = c.filterbankRate;
        c2.nBands = c.nBands;
        c2.nChannels = c.nChannels;
        return c2;
    }
    Eigen::ArrayXXf activity;

    friend BaseAlgorithm;
};