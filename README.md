# Cough Detection

## Setup:

Python 3.6

```bash
$ pip install -r requirements.txt
```

You'll need the raw audio dataset and labels dataset from FluSense saved in `flusense_audio` and `flusense_labels`.

* `loader.py` loads and preprocesses the data
* `features.py` runs feature extraction on preprocessed data
* `model.py` trains the model on extracted features

Between each step in the pipeline you need to serialize the resulting `x` and `y` arrays as `.npz` files with `np.savez_compressed`.