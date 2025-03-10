#pragma once
#include "algorithm_library/beamformer.h"
#include "framework/framework.h"

// Minimum variance distortionless response beamformer.
//
// author: Kristian Timm Andersen
class BeamformerMVDR : public AlgorithmImplementation<BeamformerConfiguration, BeamformerMVDR>
{
  public:
    BeamformerMVDR(const Coefficients &c = Coefficients()) : BaseAlgorithm{c}
    {
        filterUpdatesPerFrame = static_cast<int>(c.nBands * filterUpdateRate / c.filterbankRate);
        covarianceUpdateLambda = 1.f - expf(-1.f / (c.filterbankRate * covarianceUpdateTConstant));

        filter.resize(c.nBands, c.nChannels);
        filterNoise.resize(c.nBands, c.nChannels);
        eigenVectors.resize(c.nChannels, c.nChannels);
        Rx.resize(c.nBands);
        for (auto &rx : Rx)
        {
            rx.resize(c.nChannels, c.nChannels);
        }
        Rn.resize(c.nBands);
        for (auto &rn : Rn)
        {
            rn.resize(c.nChannels, c.nChannels);
        }
        Rxn.resize(c.nChannels, c.nChannels);
        eigenSolver = Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXcf>(c.nChannels);

        resetVariables();
        eigenSolver.compute(Rx[0], Rn[0]); // run once to make sure calculation works
    }

    enum SpeechUpdateDecisions { NOISE, SPEECH, INPUT, FREEZE_UPDATE };
    void setSpeechDecision(SpeechUpdateDecisions sd) { speechDecision = sd; }
    SpeechUpdateDecisions getSpeechDecision() const { return speechDecision; }

  private:
    void processAlgorithm(Input input, Output output)
    {
        bool activityFlag = input.signalOfInterestFlag;

        switch (speechDecision)
        {
        case SPEECH: activityFlag = true; break;
        case NOISE: activityFlag = false; break;
        default: break;
        }

        if (speechDecision != FREEZE_UPDATE) { covarianceUpdate({input.xFreq, activityFlag}); }

        for (auto i = 0; i < filterUpdatesPerFrame; i++)
        {
            calculateFilter();
            currentBand++;
            if (currentBand >= C.nBands) { currentBand = 0; }
        }

        output.yFreq = (input.xFreq * filter).rowwise().sum(); // this has been profiled to be just as fast as multiplying with ones or summing in a for-loop
        output.noiseFreq = (input.xFreq * filterNoise).rowwise().sum();
    }

    void covarianceUpdate(Input input)
    {
        for (auto band = 0; band < C.nBands; band++)
        {
            // only calculate lower triangular covariance matrix, since that is all that is used by eigensolver
            // Full matrix is: Rxn = input.xFreq.matrix().row(i).transpose() * input.xFreq.matrix().row(i).conjugate();
            for (auto channel = 0; channel < C.nChannels; channel++)
            {
                Rxn.block(channel, channel, C.nChannels - channel, 1) =
                    input.xFreq.block(band, channel, 1, C.nChannels - channel).transpose().matrix() * std::conj(input.xFreq(band, channel));
            }
            Rx[band] += covarianceUpdateLambda * (Rxn - Rx[band]);
            if (!input.signalOfInterestFlag) { Rn[band] += covarianceUpdateLambda * (Rxn - Rn[band]); }
        }
    }

    void calculateFilter()
    {
        Rx[currentBand] += 1e-16f * Eigen::MatrixXcf::Identity(C.nChannels, C.nChannels);
        Rn[currentBand] += 1e-17f * Eigen::MatrixXcf::Identity(C.nChannels, C.nChannels);
        eigenSolver.compute(Rx[currentBand], Rn[currentBand]);
        eigenVectors = eigenSolver.eigenvectors();
        Eigen::MatrixXcf invEigen = eigenVectors.inverse();

        // calculate max/min_snr beamformer with mic 0 as reference
        Eigen::VectorXcf filterMax = eigenVectors.col(C.nChannels - 1) * invEigen(C.nChannels - 1, 0);
        Eigen::VectorXcf filterMin = eigenVectors.col(0) * invEigen(C.nChannels - 1, C.nChannels - 1);

        // calculate noise power in max/min filter
        const float noisePowMaxSNR = (filterMax.adjoint() * Rn[currentBand] * filterMax)(0).real();
        const float noisePowMinSNR = (filterMin.adjoint() * Rn[currentBand] * filterMin)(0).real();

        // put resulting conjugated beamformer for current band into Filter
        filter.row(currentBand) = filterMax.adjoint();
        // scale to power in max filter
        filterNoise.row(currentBand) = filterMin.adjoint() * std::min(std::max(std::sqrt(noisePowMaxSNR / std::max(noisePowMinSNR, 1e-20f)), 1e-4f), 1e4f);
    }

    void resetVariables() final
    {
        currentBand = 0;
        for (auto i = 0u; i < Rx.size(); i++)
        {
            Rx[i] = 1e-16f * Eigen::MatrixXcf::Identity(C.nChannels, C.nChannels);
            Rx[i](0, 0) = 1e-5f;
            Rn[i] = 1e-17f * Eigen::MatrixXcf::Identity(C.nChannels, C.nChannels);
        }
        eigenVectors.setZero();
        filter.setZero();
        filter.col(0) = 1;
        filterNoise.setZero();

        Rxn.setZero();
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = filter.getDynamicMemorySize();
        size += filterNoise.getDynamicMemorySize();
        for (auto &rx : Rx)
        {
            size += rx.getDynamicMemorySize();
        }
        for (auto &rn : Rn)
        {
            size += rn.getDynamicMemorySize();
        }
        size += eigenVectors.getDynamicMemorySize();
        size += Rxn.getDynamicMemorySize();
        return size;
    }

    SpeechUpdateDecisions speechDecision = INPUT;
    static constexpr float filterUpdateRate = 4.f;          // how many times per second is the filter updated
    static constexpr float covarianceUpdateTConstant = 5.f; // covariance update smoothing time constant in seconds
    int filterUpdatesPerFrame;
    float covarianceUpdateLambda;
    int currentBand;
    Eigen::ArrayXXcf filter;
    Eigen::ArrayXXcf filterNoise;
    std::vector<Eigen::MatrixXcf> Rx, Rn;
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXcf> eigenSolver;
    Eigen::MatrixXcf eigenVectors, Rxn;

    friend BaseAlgorithm;
};