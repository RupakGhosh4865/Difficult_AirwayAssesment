# AirwayAI - Intubation Difficulty Predictor

AirwayAI is a modern, deep-learning based platform for preoperative and emergency airway assessment. It uses a ResNet18 convolutional neural network to analyze clinical photos and predict the risk of difficult intubation.

## Project Structure

- **frontend/**: Next.js application (React, Tailwind CSS, Recharts)
- **backend/**: FastAPI server (Python, PyTorch)
- **utils/**: Directory for research papers and other clinical utilities
- **data_augmented/**: Contains trained models and performance metrics

## Getting Started

### 1. Prerequisites
- Node.js (v18+)
- Python (3.8+)

### 2. Backend Setup
1. Navigate to the backend folder: `cd backend`
2. Install dependencies: `pip install -r requirements.txt`
3. Start the server: `python main.py`
   - The API will be available at `http://localhost:8000`

### 3. Frontend Setup
1. Navigate to the frontend folder: `cd frontend`
2. Install dependencies: `npm install`
3. Start the development server: `npm run dev`
   - Access the application at `http://localhost:3000`

### 4. Research Paper
- Place your research paper PDF inside the `utils/` folder. The application will automatically detect and serve it in the Research section.

## Features
- **Instant Risk Prediction**: Analyze neutral, tongue-out, and head-up photos.
- **Deep Insights**: View confidence scores and probability distributions.
- **Model Analytics**: Explore accuracy, precision, and confusion matrix data.
- **Methodology Access**: Direct access to underlying research and papers.

---

## 🚀 Deployment Guide

### Phase 1: Backend (FastAPI + AI Model) on Render
1.  **New Service**: Go to [Render.com](https://render.com) and create a new **Web Service**.
2.  **Settings**:
    *   **Root Directory**: `backend`
    *   **Runtime**: `Python 3`
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT` (⚠️ Ensure it is `main:app`, NOT `api.main:app`)
3.  **Deployment**: Click Create. Once finished, copy the URL (e.g., `https://airway-backend.onrender.com`).

### Phase 2: Frontend (Next.js) on Vercel
1.  **New Project**: Go to [Vercel.com](https://vercel.com) and import your repository.
2.  **Settings**:
    *   **Root Directory**: `frontend`
3.  **Environment Variables**:
    *   Add a variable named `NEXT_PUBLIC_API_URL`.
    *   Value: Your Render backend URL (e.g., `https://airway-backend.onrender.com`).
4.  **Deploy**: Click Deploy.

---

Developed with ❤️ by AirwayAI Team
