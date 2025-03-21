#pragma once

// helper functions
//
// author: Kristian Timm Andersen

// is x a positive power of two?
static inline bool isPositivePowerOfTwo(const int x) { return (x > 0) && ((x & (x - 1)) == 0); }

// number of bits required to represent x
static inline constexpr unsigned numberOfBits(unsigned x) { return x < 2 ? x : 1 + numberOfBits(x >> 1); }
