#include "adaptive_predictor/adaptive_predictor_kalman_time_domain.h"
#include "adaptive_predictor/adaptive_predictor_nlms_time_domain.h"

using NLMSTimeDomainImpl = Implementation<AdaptivePredictorNLMSTimeDomain, AdaptivePredictorConfiguration>;
using KalmanTimeDomainImpl = Implementation<AdaptivePredictorKalmanTimeDomain, AdaptivePredictorConfiguration>;

template <>
void Algorithm<AdaptivePredictorConfiguration>::setImplementation(const Coefficients &c)
{
    switch (c.algorithmType)
    {
    case Coefficients::KALMAN_TIME_DOMAIN: pimpl = std::make_unique<KalmanTimeDomainImpl>(c); break;
    case Coefficients::NLMS_TIME_DOMAIN:
    default:                               pimpl = std::make_unique<NLMSTimeDomainImpl>(c); break;
    }
}

AdaptivePredictor::AdaptivePredictor(const Coefficients &c) : Algorithm<AdaptivePredictorConfiguration>(c) {}
