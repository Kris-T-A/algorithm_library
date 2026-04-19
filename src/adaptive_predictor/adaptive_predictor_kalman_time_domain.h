#pragma once
#include "algorithm_library/adaptive_predictor.h"
#include "framework/framework.h"

// Time-domain random-walk Kalman adaptive line enhancer used as a periodic-component canceller.
//
//   regressor:    u[n] = [x[n - delta], x[n - delta - 1], ..., x[n - delta - N + 1]]^T
//   state model:  w[n+1] = w[n] + v[n],  v ~ N(0, Q = q*I)
//   measurement:  x[n]   = w[n]^T u[n] + r[n],  r ~ N(0, R = regularization)
//
// Per-sample update:
//   covariance += q*I               (prior)
//   yhat  = w^T u
//   e     = x - yhat                (innovation)
//   Pu    = covariance * u
//   S     = u^T Pu + R
//   K     = Pu / S
//   w    += K * e
//   covariance -= K * Pu^T          (standard rank-1 update)
//   covariance  = 0.5 (covariance + covariance^T)   (per-sample symmetrization)
//
// Initial state (construction and reset): w = 0, covariance = I (diffuse prior).
// Mapping convergenceTimeMs -> q: tauSamples = convergenceTimeMs * 1e-3 * sampleRate;
//                                 q = regularization / tauSamples^2 (clipped to [1e-20, 1e-2]).
//
// author: Kristian Timm Andersen
class AdaptivePredictorKalmanTimeDomain : public AlgorithmImplementation<AdaptivePredictorConfiguration, AdaptivePredictorKalmanTimeDomain>
{
  public:
    AdaptivePredictorKalmanTimeDomain(Coefficients c = {.algorithmType = Coefficients::KALMAN_TIME_DOMAIN}) : BaseAlgorithm{c}
    {
        const int N = c.filterLength;
        bufferSize = N + c.decorrelationDelay;
        weights.setZero(N);
        covariance.setIdentity(N, N);
        delayBuffer.setZero(bufferSize);
        u.setZero(N);
        Pu.setZero(N);
        K.setZero(N);
        writeIndex = 0;
        onParametersChanged();
    }

  private:
    void processAlgorithm(Input input, Output output)
    {
        const int N = C.filterLength;
        const int delta = C.decorrelationDelay;
        const bool outputResidual = (C.outputMode == Coefficients::RESIDUAL);
        const float R = P.regularization;
        const float q = processNoise;

        for (Eigen::Index n = 0; n < input.rows(); n++)
        {
            delayBuffer(writeIndex) = input(n);

            // Gather regressor u (length N) from delayBuffer at offsets delta ... delta + N - 1.
            for (int k = 0; k < N; k++)
            {
                int idx = writeIndex - delta - k;
                if (idx < 0) { idx += bufferSize; }
                u(k) = delayBuffer(idx);
            }

            // Prior: covariance += q * I.
            covariance.diagonal().array() += q;

            // Innovation.
            const float yhat = weights.dot(u);
            const float e = input(n) - yhat;

            // Pu = covariance * u (no-alias matrix-vector product).
            Pu.noalias() = covariance * u;

            // Innovation variance.
            const float S = u.dot(Pu) + R;

            // Kalman gain.
            K = Pu / S;

            // State update.
            weights += K * e;

            // Covariance update: covariance -= K * Pu^T (rank-1).
            covariance.noalias() -= K * Pu.transpose();

            // Per-sample symmetrization: enforce covariance = 0.5 (covariance + covariance^T).
            for (int i = 0; i < N; i++)
            {
                for (int j = i + 1; j < N; j++)
                {
                    const float avg = 0.5f * (covariance(i, j) + covariance(j, i));
                    covariance(i, j) = avg;
                    covariance(j, i) = avg;
                }
            }

            output(n) = outputResidual ? e : yhat;

            writeIndex++;
            if (writeIndex >= bufferSize) { writeIndex = 0; }
        }
    }

    void onParametersChanged()
    {
        const float tauSamples = std::max(1.f, P.convergenceTimeMs * 1e-3f * C.sampleRate);
        float q = P.regularization / (tauSamples * tauSamples);
        if (q < 1e-20f) { q = 1e-20f; }
        if (q > 1e-2f) { q = 1e-2f; }
        processNoise = q;
    }

    void resetVariables() final
    {
        weights.setZero();
        covariance.setIdentity();
        delayBuffer.setZero();
        writeIndex = 0;
    }

    size_t getDynamicSizeVariables() const final
    {
        return weights.size() * sizeof(float) + covariance.size() * sizeof(float) + delayBuffer.getDynamicMemorySize() + u.size() * sizeof(float) + Pu.size() * sizeof(float) +
               K.size() * sizeof(float);
    }

    Eigen::VectorXf weights;    // w, length N
    Eigen::MatrixXf covariance; // P, N x N
    Eigen::ArrayXf delayBuffer; // circular buffer, length N + decorrelationDelay
    Eigen::VectorXf u;          // regressor scratch, length N
    Eigen::VectorXf Pu;         // covariance * u scratch, length N
    Eigen::VectorXf K;          // Kalman gain scratch, length N

    int bufferSize;
    int writeIndex;
    float processNoise;

    friend BaseAlgorithm;
};
