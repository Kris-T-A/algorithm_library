#include "activity_detection/activity_detection_noise_estimation.h"
#include "bandsplit_downsample/bandsplit_downsample_chebyshev.h"
#include "beamformer/beamformer_mvdr.h"
#include "benchmark/benchmark.h"
#include "critical_bands/critical_bands_bark.h"
#include "dc_remover/dc_remover_first_order.h"
#include "delay/circular_buffer.h"
#include "design_iir_min_phase/design_iir_min_phase_tf2sos.h"
#include "design_iir_non_parametric/design_iir_spline.h"
#include "fft/fft_real.h"
#include "filter_min_max/filter_min_max_lemire.h"
#include "filter_power_spectrum/calculate_filter_power_spectrum.h"
#include "filterbank/filterbank_wola.h"
#include "filterbank_set/filterbank_set_wola.h"
#include "gain_calculation/gain_calculation_apriori.h"
#include "iir_filter/iir_filter_2nd_order.h"
#include "iir_filter_non_parametric/iir_filter_design_non_parametric.h"
#include "iir_filter_time_varying/state_variable_filter.h"
#include "interpolation/interpolation_cubic.h"
#include "min_phase_spectrum/min_phase_spectrum_cepstrum.h"
#include "noise_estimation/noise_estimation_activity_detection.h"
#include "noise_reduction/noise_reduction_apriori.h"
#include "noise_reduction/noise_reduction_ml.h"
#include "normal3d/normal3d_diff.h"
#include "preprocessing_path/beamformer_path.h"
#include "scale_transform/mel_scale.h"
#include "single_channel_path/noise_reduction_path.h"
#include "solver_toeplitz/solver_toeplitz_system.h"
#include "spectral_compressor/spectral_compressor_adaptive.h"
#include "spectral_compressor/spectral_compressor_wola.h"
#include "spectral_compressor/spectral_selector.h"
#include "spectrogram/spectrogram_filterbank.h"
#include "spectrogram/spectrogram_nonlinear.h"
#include "spline/spline_cubic.h"

// Macro for defining timing test using google benchmark framework
#define DEFINE_BENCHMARK_ALGORITHM(algorithm)                                                                                                                                 \
    static void algorithm##_process(benchmark::State &state)                                                                                                                  \
    {                                                                                                                                                                         \
        algorithm algo;                                                                                                                                                       \
        auto input = algo.initInput();                                                                                                                                        \
        auto output = algo.initOutput(input);                                                                                                                                 \
        for (auto _ : state)                                                                                                                                                  \
        {                                                                                                                                                                     \
            algo.process(input, output);                                                                                                                                      \
            benchmark::DoNotOptimize(algo);                                                                                                                                   \
            benchmark::DoNotOptimize(output);                                                                                                                                 \
        }                                                                                                                                                                     \
    }                                                                                                                                                                         \
    BENCHMARK(algorithm##_process);

// insert algorithms to be benchmarked. Be very careful about interpreting these results since the time depends on where in the list an algorithm is placed!
DEFINE_BENCHMARK_ALGORITHM(CircularBuffer)
DEFINE_BENCHMARK_ALGORITHM(DesignIIRMinPhaseTF2SOS)
DEFINE_BENCHMARK_ALGORITHM(DesignIIRSpline)
DEFINE_BENCHMARK_ALGORITHM(StateVariableFilter)
DEFINE_BENCHMARK_ALGORITHM(StateVariableFilterCascade)
DEFINE_BENCHMARK_ALGORITHM(IIRFilterTDFNonParametric)
DEFINE_BENCHMARK_ALGORITHM(BeamformerMVDR)
DEFINE_BENCHMARK_ALGORITHM(FilterbankSetAnalysis)
DEFINE_BENCHMARK_ALGORITHM(FilterbankSetSynthesis)
DEFINE_BENCHMARK_ALGORITHM(BeamformerPath)
DEFINE_BENCHMARK_ALGORITHM(IIRFilterCascaded)
DEFINE_BENCHMARK_ALGORITHM(NoiseEstimationActivityDetection)
DEFINE_BENCHMARK_ALGORITHM(SplineCubic)
DEFINE_BENCHMARK_ALGORITHM(InterpolationCubicSample)
DEFINE_BENCHMARK_ALGORITHM(InterpolationCubic)
DEFINE_BENCHMARK_ALGORITHM(InterpolationCubicConstant)
DEFINE_BENCHMARK_ALGORITHM(FFTReal)
DEFINE_BENCHMARK_ALGORITHM(IIRFilter2ndOrder)
DEFINE_BENCHMARK_ALGORITHM(SolverToeplitzSystem)
DEFINE_BENCHMARK_ALGORITHM(FilterbankAnalysisWOLA)
DEFINE_BENCHMARK_ALGORITHM(FilterbankSynthesisWOLA)
DEFINE_BENCHMARK_ALGORITHM(SpectrogramFilterbank)
DEFINE_BENCHMARK_ALGORITHM(SpectrogramNonlinear)
DEFINE_BENCHMARK_ALGORITHM(Normal3dDiff)
DEFINE_BENCHMARK_ALGORITHM(MinPhaseSpectrumCepstrum)
DEFINE_BENCHMARK_ALGORITHM(CriticalBandsBarkSum)
DEFINE_BENCHMARK_ALGORITHM(FilterMinMaxLemire)
DEFINE_BENCHMARK_ALGORITHM(FilterMaxLemire)
DEFINE_BENCHMARK_ALGORITHM(FilterMinLemire)
DEFINE_BENCHMARK_ALGORITHM(StreamingMinMaxLemire)
DEFINE_BENCHMARK_ALGORITHM(StreamingMaxLemire)
DEFINE_BENCHMARK_ALGORITHM(StreamingMinLemire)
DEFINE_BENCHMARK_ALGORITHM(DCRemoverFirstOrder)
DEFINE_BENCHMARK_ALGORITHM(MelScale)
DEFINE_BENCHMARK_ALGORITHM(ActivityDetectionNoiseEstimation)
DEFINE_BENCHMARK_ALGORITHM(ActivityDetectionFusedNoiseEstimation)
DEFINE_BENCHMARK_ALGORITHM(GainCalculationApriori)
DEFINE_BENCHMARK_ALGORITHM(CalculateFilterPowerSpectrum)
DEFINE_BENCHMARK_ALGORITHM(NoiseReductionPath)
DEFINE_BENCHMARK_ALGORITHM(NoiseReductionAPriori)
DEFINE_BENCHMARK_ALGORITHM(NoiseReductionML)
DEFINE_BENCHMARK_ALGORITHM(BandsplitDownsampleChebyshev)
DEFINE_BENCHMARK_ALGORITHM(CombineBandsplitDownsampleChebyshev)
DEFINE_BENCHMARK_ALGORITHM(SpectralCompressorWOLA)
DEFINE_BENCHMARK_ALGORITHM(SpectralCompressorAdaptive)
DEFINE_BENCHMARK_ALGORITHM(SpectralSelector)

// benchmark inverse FFT
static void FFTInverse_process(benchmark::State &state)
{
    FFTReal algo;
    auto input = algo.initInput();
    auto output = algo.initOutput(input);
    for (auto _ : state)
    {
        algo.inverse(output, input);
        benchmark::DoNotOptimize(algo);
        benchmark::DoNotOptimize(input);
    }
}
BENCHMARK(FFTInverse_process);

// main function
BENCHMARK_MAIN();
