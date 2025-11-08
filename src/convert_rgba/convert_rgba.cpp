#include "convert_rgba/convert_rgba_variations.h"

using OceanImpl = Implementation<ConvertRGBAOcean, ConvertRGBAConfiguration>;
using ParulaImpl = Implementation<ConvertRGBAParula, ConvertRGBAConfiguration>;
using ViridisImpl = Implementation<ConvertRGBAViridis, ConvertRGBAConfiguration>;
using PlasmaImpl = Implementation<ConvertRGBAPlasma, ConvertRGBAConfiguration>;
using MagmaImpl = Implementation<ConvertRGBAMagma, ConvertRGBAConfiguration>;

template <>
void Algorithm<ConvertRGBAConfiguration>::setImplementation(const Coefficients &c)
{
    switch (c.colorScale)
    {
    case ConvertRGBAConfiguration::Coefficients::OCEAN: pimpl = std::make_unique<OceanImpl>(c); break;
    default:
    case ConvertRGBAConfiguration::Coefficients::PARULA: pimpl = std::make_unique<ParulaImpl>(c); break;
    case ConvertRGBAConfiguration::Coefficients::VIRIDIS: pimpl = std::make_unique<ViridisImpl>(c); break;
    case ConvertRGBAConfiguration::Coefficients::MAGMA: pimpl = std::make_unique<MagmaImpl>(c); break;
    case ConvertRGBAConfiguration::Coefficients::PLASMA: pimpl = std::make_unique<PlasmaImpl>(c); break;
    }
}

ConvertRGBA::ConvertRGBA(const Coefficients &c) : Algorithm<ConvertRGBAConfiguration>(c) {}
