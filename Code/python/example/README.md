# Example Smoke Test

This folder contains `main.py`, a small script for comparing the original Python EzGATr model with the ASL/C++ implementation.

## Run

From the `Code/python/` directory:

```bash
python3 example/main.py
```

## Build First

Before running the ASL/C++ model, compile the extensions:

```bash
python3 setup.py build_ext
```

If basis files are missing:

```bash
python3 -m ezgatr_extensions.make_basis
python3 setup.py build_ext
```

## Output

The script prints:

- input size
- model creation steps
- runtime for Python and ASL/C++
- output shapes
- maximum absolute difference



## Custom Input Size

```bash
python3 example/main.py -s 1,4,2
```

Format:

```text
batch,tokens,channels
```

## Help

```bash
python3 example/main.py --help
```
