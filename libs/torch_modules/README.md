# torch_modules

PyTorch ports of algorithms from `algorithm_library/src/`. Designed for training-time feature extraction (autograd, GPU) and inference parity with the C++ implementations.

## Install (editable)

```bash
pip install -e libs/torch_modules[test]
```

## Run tests

```bash
pytest libs/torch_modules/tests
```

Alignment tests need the `python_algorithm_library` pyd built (see that library's README). Edge-case tests run without it.
