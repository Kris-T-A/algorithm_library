#include "adaptive_predictor/adaptive_predictor_kalman_freq_domain.h"
#include "adaptive_predictor/adaptive_predictor_kalman_time_domain.h"
#include "adaptive_predictor/adaptive_predictor_nlms_freq_domain.h"
#include "adaptive_predictor/adaptive_predictor_nlms_moment.h"
#include "adaptive_predictor/adaptive_predictor_nlms_moment_time_domain.h"
#include "adaptive_predictor/adaptive_predictor_nlms_time_domain.h"

using NLMSTimeDomainImpl = Implementation<AdaptivePredictorNLMSTimeDomain, AdaptivePredictorConfiguration>;
using KalmanTimeDomainImpl = Implementation<AdaptivePredictorKalmanTimeDomain, AdaptivePredictorConfiguration>;
using NLMSFreqDomainImpl = Implementation<AdaptivePredictorNLMSFreqDomain, AdaptivePredictorConfiguration>;
using KalmanFreqDomainImpl = Implementation<AdaptivePredictorKalmanFreqDomain, AdaptivePredictorConfiguration>;
using NLMSMomentumImpl = Implementation<AdaptivePredictorNLMSMomentum, AdaptivePredictorConfiguration>;
using NLMSMomentumTimeDomainImpl = Implementation<AdaptivePredictorNLMSMomentumTimeDomain, AdaptivePredictorConfiguration>;

template <>
void Algorithm<AdaptivePredictorConfiguration>::setImplementation(const Coefficients &c)
{
    switch (c.algorithmType)
    {
    case Coefficients::KALMAN_TIME_DOMAIN:        pimpl = std::make_unique<KalmanTimeDomainImpl>(c); break;
    case Coefficients::NLMS_FREQ_DOMAIN:          pimpl = std::make_unique<NLMSFreqDomainImpl>(c); break;
    case Coefficients::KALMAN_FREQ_DOMAIN:        pimpl = std::make_unique<KalmanFreqDomainImpl>(c); break;
    case Coefficients::NLMS_MOMENTUM:             pimpl = std::make_unique<NLMSMomentumImpl>(c); break;
    case Coefficients::NLMS_MOMENTUM_TIME_DOMAIN: pimpl = std::make_unique<NLMSMomentumTimeDomainImpl>(c); break;
    case Coefficients::NLMS_TIME_DOMAIN:
    default:                                      pimpl = std::make_unique<NLMSTimeDomainImpl>(c); break;
    }
}

AdaptivePredictor::AdaptivePredictor(const Coefficients &c) : Algorithm<AdaptivePredictorConfiguration>(c) {}
