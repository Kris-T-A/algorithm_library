// #include "adaptive_predictor/adaptive_predictor_kalman_freq_domain.h"
// #include "adaptive_predictor/adaptive_predictor_kalman_time_domain.h"
// #include "adaptive_predictor/adaptive_predictor_nlms_freq_domain.h"
// #include "adaptive_predictor/adaptive_predictor_nlms_moment.h"
// #include "adaptive_predictor/adaptive_predictor_nlms_moment_time_domain.h"
// #include "adaptive_predictor/adaptive_predictor_nlms_time_domain.h"
// #include "unit_test.h"
// #include "gtest/gtest.h"
// #include <chrono>

// using namespace Eigen;

// namespace
// {

// ArrayXf makeNoisySinusoid(int nSamples, float sampleRate, float freqHz, float toneAmp, float noiseAmp, unsigned seed)
// {
//     srand(seed);
//     ArrayXf noise = ArrayXf::Random(nSamples) * noiseAmp;
//     ArrayXf t = ArrayXf::LinSpaced(nSamples, 0.f, static_cast<float>(nSamples - 1) / sampleRate);
//     ArrayXf tone = (2.f * static_cast<float>(M_PI) * freqHz * t).sin() * toneAmp;
//     return tone + noise;
// }

// ArrayXf makeStepFrequencySinusoid(int nSamples, float sampleRate, float f1, float f2, int nStep, float toneAmp, float noiseAmp, unsigned seed)
// {
//     srand(seed);
//     ArrayXf noise = ArrayXf::Random(nSamples) * noiseAmp;
//     ArrayXf signal(nSamples);
//     double phase = 0.0;
//     const double twoPi = 2.0 * 3.14159265358979323846;
//     for (int n = 0; n < nSamples; n++)
//     {
//         const double f = (n < nStep) ? static_cast<double>(f1) : static_cast<double>(f2);
//         signal(n) = toneAmp * static_cast<float>(std::sin(phase)) + noise(n);
//         phase += twoPi * f / static_cast<double>(sampleRate);
//         if (phase > twoPi) { phase -= twoPi; }
//     }
//     return signal;
// }

// struct Metrics
// {
//     float steadyResidualPower;      // tail residual after 1 s on stationary tone + noise
//     float preStepResidualPower;     // steady-state residual before the frequency step
//     float postStepResidualPower;    // steady-state residual after re-adapting to new frequency
//     float slowSettingResidualPower; // post-step residual with a very-slow convergenceTimeMs
//     size_t dynamicMemory;
//     double nsPerSample;             // best-of-N wall-clock nanoseconds per input sample
// };

// // Run the algorithm twice (once on a stationary signal, twice on a step-frequency signal with
// // fast and slow convergenceTimeMs). The three residual-power measurements isolate:
// //   - steady-state suppression (converges?)
// //   - tracking through a frequency step (nonstationary performance)
// //   - effect of the convergenceTimeMs knob (slow setting should leave a larger residual)
// template <typename Impl>
// Metrics measureVariant(const typename Impl::Coefficients &baseCoeffs, float fastConvergenceMs, float slowConvergenceMs)
// {
//     const float fs = baseCoeffs.sampleRate;
//     Metrics m{};

//     // Steady-state suppression of a stationary tone. Also used to time per-sample cost.
//     {
//         const int nSamples = 48000; // 1s
//         typename Impl::Coefficients c = baseCoeffs;
//         c.outputMode = Impl::Coefficients::RESIDUAL;
//         Impl algo(c);
//         auto p = algo.getParameters();
//         p.convergenceTimeMs = fastConvergenceMs;
//         algo.setParameters(p);

//         ArrayXf input = makeNoisySinusoid(nSamples, fs, 1000.f, 1.f, 0.1f, 0);
//         ArrayXf output(nSamples);
//         algo.process(input, output);

//         m.steadyResidualPower = output.tail(nSamples / 4).square().mean();
//         m.dynamicMemory = algo.getDynamicSize();

//         // Best-of-N timing: reruns the process call on the already-converged filter so we get
//         // steady-state per-sample cost without startup transients dominating the measurement.
//         constexpr int kTimingReps = 5;
//         double bestNs = std::numeric_limits<double>::infinity();
//         for (int rep = 0; rep < kTimingReps; rep++)
//         {
//             const auto t0 = std::chrono::steady_clock::now();
//             algo.process(input, output);
//             const auto t1 = std::chrono::steady_clock::now();
//             const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
//             if (ns < bestNs) { bestNs = ns; }
//         }
//         m.nsPerSample = bestNs / static_cast<double>(nSamples);
//     }

//     // Frequency-step tracking at the fast convergenceTimeMs setting.
//     {
//         const int nSamples = 96000; // 2s total
//         const int nStep = 48000;    // step at 1s
//         typename Impl::Coefficients c = baseCoeffs;
//         c.outputMode = Impl::Coefficients::RESIDUAL;
//         Impl algo(c);
//         auto p = algo.getParameters();
//         p.convergenceTimeMs = fastConvergenceMs;
//         algo.setParameters(p);

//         ArrayXf input = makeStepFrequencySinusoid(nSamples, fs, 1000.f, 1200.f, nStep, 1.f, 0.05f, 0);
//         ArrayXf output(nSamples);
//         algo.process(input, output);

//         const int windowLen = nSamples / 4;
//         m.preStepResidualPower = output.segment(nStep - windowLen, windowLen).square().mean();
//         m.postStepResidualPower = output.segment(nSamples - windowLen, windowLen).square().mean();
//     }

//     // Same step signal but with a very slow convergenceTimeMs — residual should stay large after
//     // the step because the algorithm can't re-adapt within the window.
//     {
//         const int nSamples = 96000;
//         const int nStep = 48000;
//         typename Impl::Coefficients c = baseCoeffs;
//         c.outputMode = Impl::Coefficients::RESIDUAL;
//         Impl algo(c);
//         auto p = algo.getParameters();
//         p.convergenceTimeMs = slowConvergenceMs;
//         algo.setParameters(p);

//         ArrayXf input = makeStepFrequencySinusoid(nSamples, fs, 1000.f, 1200.f, nStep, 1.f, 0.05f, 0);
//         ArrayXf output(nSamples);
//         algo.process(input, output);

//         const int windowLen = nSamples / 4;
//         m.slowSettingResidualPower = output.segment(nSamples - windowLen, windowLen).square().mean();
//     }

//     return m;
// }

// float powerToDb(float power) { return 10.f * std::log10(std::max(power, 1e-20f)); }

// void printRow(const char *name, int filterLength, const Metrics &m)
// {
//     fmt::print("| {:<28} | {:>4} | {:>7.1f} dB | {:>7.1f} dB | {:>7.1f} dB | {:>7.1f} dB | {:>10} | {:>8.1f} |\n", name, filterLength,
//                powerToDb(m.steadyResidualPower), powerToDb(m.preStepResidualPower), powerToDb(m.postStepResidualPower),
//                powerToDb(m.slowSettingResidualPower), m.dynamicMemory, m.nsPerSample);
// }

// } // namespace

// // Side-by-side comparison of the three AdaptivePredictor variants on identical stimulus.
// // Columns report algorithmic quality (residual power after adaptation) under different conditions:
// //   - Steady   : residual on a stationary 1 kHz + noise input, fast convergenceTimeMs.
// //   - Pre      : residual just before a 1 kHz -> 1.2 kHz frequency step, fast convergenceTimeMs.
// //   - Post     : residual after re-adapting to the step, fast convergenceTimeMs. Post close to Pre
// //                means the algorithm has re-converged within the recovery window.
// //   - Slow     : residual after the step with a very-slow convergenceTimeMs. Should be much higher
// //                than Post if the convergenceTimeMs knob actually slows down adaptation.
// //   - Dyn bytes: dynamic memory footprint.
// //   - ns/samp  : best-of-N wall-clock nanoseconds per input sample, timed on the stationary
// //                signal after the filter has converged. Captures steady-state per-sample cost.
// // All variants run at filterLength = 64. For the time-domain variants this is the number of FIR
// // taps; for the freq-domain variants it's the fftSize (so nBands = 33, hop = 16). Each freq-domain
// // bin still carries kFilterLength=4 complex taps internally, so their total modeling capacity
// // (~33*4 complex = 264 real coefficients) already exceeds the 64-tap time-domain filters —
// // matching fftSize to the time-domain N is the fair operating point, not doubling it.
// TEST(AdaptivePredictorComparison, PredictionQualityTable)
// {
//     const float fs = 48000.f;
//     const float fastConvergenceMs = 20.f;
//     const float slowConvergenceMs = 5000.f;

//     fmt::print("\n");
//     fmt::print("| {:<28} | {:>4} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8} |\n", "Variant", "Len", "Steady", "Pre", "Post", "Slow",
//                "Dyn bytes", "ns/samp");
//     fmt::print("|{:-<30}|{:->6}|{:->12}|{:->12}|{:->12}|{:->12}|{:->12}|{:->10}|\n", "", "", "", "", "", "", "", "");

//     AdaptivePredictorConfiguration::Coefficients c;
//     c.sampleRate = fs;
//     c.decorrelationDelay = 1;

//     auto mNLMS = measureVariant<AdaptivePredictorNLMSTimeDomain>(c, fastConvergenceMs, slowConvergenceMs);
//     printRow("NLMS time-domain", c.filterLength, mNLMS);

//     auto mKT = measureVariant<AdaptivePredictorKalmanTimeDomain>(c, fastConvergenceMs, slowConvergenceMs);
//     printRow("Kalman time-domain", c.filterLength, mKT);

//     auto mNMT = measureVariant<AdaptivePredictorNLMSMomentumTimeDomain>(c, fastConvergenceMs, slowConvergenceMs);
//     printRow("NLMS momentum time-domain", c.filterLength, mNMT);

//     auto mNF = measureVariant<AdaptivePredictorNLMSFreqDomain>(c, fastConvergenceMs, slowConvergenceMs);
//     printRow("NLMS freq-domain (WOLA)", c.filterLength, mNF);

//     auto mNM = measureVariant<AdaptivePredictorNLMSMomentum>(c, fastConvergenceMs, slowConvergenceMs);
//     printRow("NLMS momentum (WOLA)", c.filterLength, mNM);

//     auto mKF = measureVariant<AdaptivePredictorKalmanFreqDomain>(c, fastConvergenceMs, slowConvergenceMs);
//     printRow("Kalman freq-domain (WOLA)", c.filterLength, mKF);

//     fmt::print("\n");

//     // Sanity: each variant must meaningfully suppress the steady-state tone.
//     // Input tone power is ~0.5 (sinusoid at amplitude 1); expect at least 10x reduction.
//     for (const auto &pair : {std::make_pair("NLMS", mNLMS), std::make_pair("Kalman-time", mKT), std::make_pair("NLMS-momentum-time", mNMT),
//                              std::make_pair("NLMS-freq", mNF), std::make_pair("NLMS-momentum", mNM), std::make_pair("Kalman-freq", mKF)})
//     {
//         EXPECT_LT(pair.second.steadyResidualPower, 0.05f) << pair.first << " failed steady-state tone suppression";
//     }
// }
