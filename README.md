# 🩺 Intubation Difficulty Predictor (AI-Assisted)

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An AI-powered application for preoperative and emergency airway assessment. Upload clinical photos and get instant risk predictions—**Easy** or **Difficult**—with confidence scores and full transparency.

## 🎯 Features

- Real-time airway difficulty prediction
- Support for multiple image analysis (Neutral, Tongue-out, Head-up positions)
- Confidence scores for predictions
- Built on ResNet18 architecture
- Interactive web interface
- Educational resources about intubation
- Comprehensive visualization of results

## 🏗️ Project Structure

```
airway/
├── app.py                  # Main Streamlit application
├── train_intubation.py     # Model training script
├── data/                   # Original dataset
│   ├── class_names.txt     # Class labels
│   ├── model_intubation.pt # Trained model
│   ├── difficult/          # Difficult intubation images
│   └── easy/              # Easy intubation images
├── data_augmented/         # Augmented dataset
│   ├── class_names.txt
│   ├── model_intubation.pt
│   ├── difficult/
│   └── easy/
└── images/                 # Example images for UI
    ├── 4.jpg
    ├── 5.jpg
    └── 6.jpg
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RupakGhosh4865/Difficult_AirwayAssesment.git
   cd Difficult_AirwayAssesment
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the Streamlit server:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

## 📸 Image Requirements

For accurate predictions, provide three standardized photos:

1. **Neutral Position**
   - Facing camera
   - Mouth closed
   - Full face visible

2. **Tongue Extended**
   - Facing camera
   - Mouth open
   - Tongue fully extended

3. **Head Up (Sniffing Position)**
   - Side view
   - Neck extended
   - Full profile visible

## 🧠 Model Details

- **Architecture**: ResNet18
- **Input Size**: 224x224 pixels
- **Output**: Binary classification (Easy/Difficult)
- **Training Parameters**:
  - Batch size: 8
  - Epochs: 10
  - Optimizer: Adam
  - Learning rate: 1e-4

### Data Augmentation Techniques

- RandomResizedCrop (scale 0.7–1.0)
- RandomHorizontalFlip
- ColorJitter
- RandomRotation (±15 degrees)

## ⚠️ Disclaimer

This application is designed for **educational and training purposes only**. It should not be used as the sole basis for clinical decision-making. Always rely on comprehensive clinical assessment and professional judgment for actual patient care.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Rupak Ghosh** - *Initial work* - [RupakGhosh4865](https://github.com/RupakGhosh4865)

## 🙏 Acknowledgments

- ResNet architecture by Microsoft Research
- Streamlit for the amazing web framework
- PyTorch team for the deep learning framework

---
Made with ❤️ for the medical community
