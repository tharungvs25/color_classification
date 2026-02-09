# Quick Start Guide - Color Classification Project

This guide will get you up and running in under 5 minutes!

## 🚀 Quick Setup (Windows)

### 1. Create & Activate Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## 📊 Run the Complete Pipeline

Execute these commands in order:

### Step 1: Load Dataset
```bash
python load_dataset.py
```
⏱️ Time: ~30-60 seconds (downloads dataset)

### Step 2: Explore Data (Optional but Recommended)
```bash
python explore_data.py
```
⏱️ Time: ~5-10 seconds

### Step 3: Preprocess
```bash
python preprocess.py
```
⏱️ Time: ~2-5 seconds

### Step 4: Train Model
```bash
python train_model.py
```
⏱️ Time: ~10-30 seconds

### Step 5: Test Predictions
```bash
python predict_color.py test
```
⏱️ Time: ~1 second

### Step 6: Real-Time Detection (with webcam)
```bash
python webcam_detection.py
```
Press 'q' to quit

## ⚡ All-in-One Script

Want to run everything at once? Create a file called `run_all.py`:

```python
import subprocess
import sys

scripts = [
    'load_dataset.py',
    'explore_data.py',
    'preprocess.py',
    'train_model.py',
    'predict_color.py test',
]

for script in scripts:
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable] + script.split(), capture_output=False)
    if result.returncode != 0:
        print(f"\n❌ Error running {script}")
        break

print("\n✅ Pipeline complete! Now try: python webcam_detection.py")
```

Then run:
```bash
python run_all.py
```

## 🧪 Quick Test

Test a single color prediction:
```bash
python predict_color.py 255 0 0
```
Expected output: `RGB(255, 0, 0) → RED`

## 📋 Troubleshooting

### Issue: "mlcroissant not found"
```bash
pip install mlcroissant
```

### Issue: "Cannot open camera"
- Close other applications using the webcam
- Try different camera ID: Edit `webcam_detection.py`, change `camera_id=0` to `camera_id=1`

### Issue: "Model not found"
Make sure you ran `train_model.py` first!

## 📂 Expected Output Files

After running the pipeline, you should have:

```
color classification manual/
├── color_dataset.csv                    # Downloaded dataset
├── rgb_distribution.png                 # Visualization
├── preprocessed_data/                   # Preprocessed data
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_test.npy
│   └── label_encoder.pkl
└── models/                              # Trained model
    └── random_forest_color_classifier.pkl
```

## 🎯 What to Show in Interviews

1. **Run webcam detection** - Most impressive visual demo
2. **Explain the pipeline** - Show understanding of ML workflow
3. **Discuss model choice** - Why Random Forest?
4. **Show code quality** - Clean, documented, modular

## 📖 Next Steps

- Read the full [README.md](README.md) for detailed explanations
- Experiment with different RGB values
- Try modifying hyperparameters in [train_model.py](train_model.py)
- Add new features or improve the model

---

**Total time from setup to working demo: ~5-10 minutes** ⚡
