# Sign Translate

Prototype for a sign language translation application.

The current implementation is **Phase 1**: a local Python prototype that recognizes a small set of single-hand gestures from webcam video using MediaPipe hand landmarks and a scikit-learn classifier.

## Phase 1 Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Collect samples for at least two labels:

```bash
python3 phase1_sign_recognition/collect_dataset.py --label hello --samples 120
python3 phase1_sign_recognition/collect_dataset.py --label thanks --samples 120
```

Train the first classifier:

```bash
python3 phase1_sign_recognition/train_model.py
```

Run realtime prediction:

```bash
python3 phase1_sign_recognition/predict_realtime.py
```

See [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) for the full project notes and next-phase plan.

## Using TSL-51

After cloning or downloading TSL-51, inspect the local folder first:

```bash
python3 phase1_sign_recognition/inspect_tsl51.py /path/to/thai-sign-language-tsl51
```

Convert landmark files into the Phase 1 CSV format:

```bash
python3 phase1_sign_recognition/convert_tsl51_landmarks.py /path/to/thai-sign-language-tsl51 --output data/raw/tsl51_landmarks.csv
python3 phase1_sign_recognition/train_model.py --input data/raw/tsl51_landmarks.csv --model models/tsl51_sign_classifier.joblib
```

The TSL-51 converter reads `landmarks/*_sign/*.csv` and uses
`metadata/user_sign_metadata.csv` for clean labels. It currently exports one
hand per frame (right hand when available, otherwise left hand); two-hand
signs are not yet supported by the realtime one-hand model.

Run the automated checks with:

```bash
python3 -m unittest discover -s tests -v
```
