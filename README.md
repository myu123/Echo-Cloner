# Echo Cloner - XTTS-v2

A modern, full-stack voice cloning application using XTTS-v2 for high-quality voice synthesis. Upload audio, train custom voice models, and generate speech with your cloned voice.

## Features

- **Audio Upload & Processing**: Upload long-form audio (podcasts, interviews) with drag-and-drop support
- **Automatic Segmentation**: Whisper-powered transcription with intelligent audio segmentation
- **GPU-Accelerated Training**: XTTS-v2 fine-tuning with real-time progress tracking via WebSocket
- **Model Management**: Save, load, and manage multiple trained voice models
- **Text-to-Speech Generation**: Generate natural-sounding speech from trained models
- **Modern UI**: React-based dark-mode interface with responsive design

## Tech Stack

### Frontend
- **React 18** with TypeScript
- **Tailwind CSS** for styling
- **React Router** for navigation
- **Axios** for API calls
- **WebSocket** for real-time training updates
- **Vite** for fast development and building

### Backend
- **FastAPI** for REST API
- **PyTorch** with CUDA support
- **Coqui TTS (XTTS-v2)** for voice synthesis and cloning
- **OpenAI Whisper** for transcription and segmentation
- **WebSocket** for real-time progress updates

## Prerequisites

### System Requirements
- **OS**: Windows 11 (WSL2), Linux, or macOS
- **GPU**: NVIDIA GPU with 12GB+ VRAM (RTX 3060/4060 or better)
- **RAM**: 16GB+ recommended
- **Disk Space**: 20GB+ free space

### Software Requirements
- **Python**: 3.10 or 3.11 (REQUIRED - 3.13 NOT supported by Coqui TTS)
- **Node.js**: 18 or higher
- **CUDA**: 12.1 or compatible version
- **Git**: For cloning the repository

⚠️ **IMPORTANT**: You MUST use Python 3.10 or 3.11. Python 3.13 will NOT work because Coqui TTS doesn't support it yet.

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd voice-cloning-app
```

### 2. Backend Setup

#### Check Python Version First

```bash
python --version
# Should show Python 3.10.x or 3.11.x
# If you see 3.13 or higher, you need to install Python 3.11
```

**If you have Python 3.13:**
- Download Python 3.11 from [python.org](https://www.python.org/downloads/)
- On Windows with multiple Python versions installed, use: `py -3.11 -m venv venv`
- On Linux/WSL: Install python3.11 package, then use `python3.11 -m venv venv`

#### Install Python Dependencies

```bash
cd backend

# Create virtual environment with Python 3.11 (Windows with multiple versions)
py -3.11 -m venv venv

# OR use default python if it's already 3.11
python -m venv venv

# Activate virtual environment
# On Windows PowerShell:
venv\Scripts\activate

# On Windows/WSL/Linux:
source venv/bin/activate

# Verify Python version in venv
python --version  # MUST show 3.10.x or 3.11.x

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Install PyTorch with CUDA Support

```bash
# For CUDA 12.1 (adjust version as needed)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Verify CUDA Installation

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected output:
```
CUDA Available: True
CUDA Version: 12.1
GPU: NVIDIA GeForce RTX 3060
```

### 3. Frontend Setup

#### Install Node Dependencies

```bash
cd ../frontend
npm install
```

## Running the Application

You need to run both the backend and frontend servers.

### Terminal 1: Start Backend

```bash
cd backend
source venv/bin/activate  # Activate virtual environment
python run.py
```

The backend API will start on `http://localhost:8000`

- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/system/health

### Terminal 2: Start Frontend

```bash
cd frontend
npm run dev
```

The frontend will start on `http://localhost:3000`

Visit http://localhost:3000 in your browser to use the application.

## Usage Workflow

### 1. Upload Audio Files
1. Navigate to the **Upload** page
2. Drag and drop audio files (MP3, WAV, M4A, OGG, FLAC, AAC)
3. Upload 15-30 minutes of audio for best results (minimum 10 minutes)
4. Click "Process with Whisper" to transcribe and segment audio

### 2. Train a Model
1. Navigate to the **Train** page
2. Review dataset validation (shows total duration and segment count)
3. Enter a model name (e.g., "My Voice Model")
4. Configure training parameters:
   - **Epochs**: 15 (recommended, range: 5-30)
   - **Batch Size**: 2 (recommended for 12GB VRAM)
   - **Learning Rate**: 5e-6 (default)
5. Click "Start Training"
6. Monitor real-time progress (loss, GPU usage, time remaining)
7. Training completes automatically and saves the model

### 3. Generate Speech
1. Navigate to the **Generate** page
2. Select a trained model from the dropdown
3. Enter text to speak (10-200 characters recommended)
4. Adjust temperature and speed if desired
5. Click "Generate Speech"
6. Play or download the generated audio

### 4. Manage Models
1. Navigate to the **Models** page
2. View all trained models with metadata
3. Actions available:
   - **Use Model**: Navigate to generation page
   - **Test**: Quick test with sample text
   - **Rename**: Edit model name inline
   - **Delete**: Remove model and files

## Configuration

### Backend Configuration

Edit `backend/app/models/config.py` for advanced settings:

```python
# Audio Processing
SAMPLE_RATE = 22050
TARGET_SEGMENT_LENGTH_MIN = 3.0  # seconds
TARGET_SEGMENT_LENGTH_MAX = 10.0  # seconds
MIN_TOTAL_AUDIO_DURATION = 600.0  # 10 minutes

# Training
DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 5e-6
```

### Environment Variables

Create a `.env` file in the `backend` directory:

```env
API_HOST=0.0.0.0
API_PORT=8000
WHISPER_MODEL=large-v2
CORS_ORIGINS=["http://localhost:3000"]
```

## Troubleshooting

### Python Version Error / TTS Installation Failed

**Problem**: `ERROR: No matching distribution found for TTS` or version compatibility errors

**Cause**: You're using Python 3.13 (or 3.12), which is not supported by Coqui TTS

**Solution**:
1. **Download Python 3.11** from [python.org](https://www.python.org/downloads/)
2. **Delete your current venv**:
   ```bash
   # Deactivate if active
   deactivate
   # Delete venv folder
   rm -rf venv  # Linux/WSL
   rmdir /s venv  # Windows CMD
   ```
3. **Create new venv with Python 3.11**:
   ```bash
   # Windows (with multiple Python versions)
   py -3.11 -m venv venv

   # Linux/WSL
   python3.11 -m venv venv
   ```
4. **Activate and verify**:
   ```bash
   venv\Scripts\activate  # Windows
   python --version  # Should show 3.11.x
   ```
5. **Install requirements again**:
   ```bash
   pip install -r requirements.txt
   ```

### CUDA Not Available

**Problem**: "CUDA not available" warning on startup

**Solutions**:
1. Verify NVIDIA drivers are installed: `nvidia-smi`
2. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchaudio
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
3. Check CUDA version compatibility with PyTorch

### Out of Memory (OOM) Errors

**Problem**: GPU runs out of memory during training

**Solutions**:
1. Reduce batch size to 1
2. Reduce number of audio segments
3. Close other GPU-intensive applications
4. Use gradient accumulation (already set to 2)

### Transcription Fails

**Problem**: Whisper transcription errors or crashes

**Solutions**:
1. Ensure audio files are valid (not corrupted)
2. Check file format is supported
3. Reduce file size if needed
4. Clear cache: `rm -rf backend/cache/*`

### WebSocket Connection Errors

**Problem**: Training progress not updating

**Solutions**:
1. Check both backend and frontend are running
2. Verify WebSocket URL in browser console
3. Refresh the page and reconnect

### Port Already in Use

**Problem**: "Address already in use" error

**Solutions**:
```bash
# Find process using port 8000
lsof -i :8000  # On Linux/Mac
netstat -ano | findstr :8000  # On Windows

# Kill the process or use a different port
```

## Project Structure

```
voice-cloning-app/
├── backend/
│   ├── app/
│   │   ├── api/          # API routes and WebSocket
│   │   ├── services/     # ML services (Whisper, XTTS, etc.)
│   │   ├── models/       # Pydantic schemas and config
│   │   ├── utils/        # Utility functions
│   │   └── main.py       # FastAPI app entry point
│   ├── data/             # User data (uploads, processed, datasets)
│   ├── trained_models/   # Saved XTTS models
│   ├── cache/            # Whisper model cache
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── pages/        # Page components
│   │   ├── services/     # API and WebSocket services
│   │   ├── types/        # TypeScript types
│   │   ├── utils/        # Helper functions
│   │   └── App.tsx       # Main app component
│   └── package.json
└── README.md
```

## API Documentation

Once the backend is running, visit http://localhost:8000/docs for interactive API documentation (Swagger UI).

### Key Endpoints

- `POST /api/upload` - Upload audio files
- `POST /api/transcribe` - Transcribe and segment audio
- `POST /api/train` - Start model training
- `GET /api/train/status` - Get training status
- `POST /api/generate` - Generate speech
- `GET /api/models` - List all models
- `WS /ws/training` - WebSocket for real-time training updates

## Clearing All Data

If you want to reset the application to a fresh state (delete all uploads, processed audio, datasets, and trained models):

```bash
# Run from the project root directory
python clear_data.py
```

This will:
- Ask for confirmation (type `yes` to proceed)
- Delete all uploaded audio files
- Delete all processed/segmented audio
- Delete all training datasets
- Delete all trained models
- Delete generated audio and cache files

⚠️ **Warning**: This action cannot be undone!

## Performance Tips

1. **Audio Quality**: Use high-quality audio (48kHz, WAV preferred)
2. **Training Duration**: 20-30 minutes of audio gives best results
3. **Batch Size**: Use 2 for 12GB VRAM, 4 for 24GB VRAM
4. **Epochs**: Start with 15, increase to 25 for higher quality
5. **GPU Memory**: Close other applications to free VRAM

## Known Limitations

- Requires NVIDIA GPU with CUDA support (no CPU-only training)
- Training time: ~1-2 hours for 15 epochs on 20 mins of audio
- Best results with clean, clear speech audio
- Model file sizes: ~1-2GB per trained model

## License

This project uses:
- **Coqui TTS** (Mozilla Public License 2.0)
- **OpenAI Whisper** (MIT License)
- Other open-source libraries (see requirements.txt and package.json)

## Support

For issues, questions, or contributions:
1. Check the Troubleshooting section
2. Review API docs at http://localhost:8000/docs
3. Check console logs (backend terminal and browser DevTools)

## Acknowledgments

- [Coqui TTS](https://github.com/coqui-ai/TTS) for XTTS-v2
- [OpenAI Whisper](https://github.com/openai/whisper) for transcription
- FastAPI, React, and the open-source community

---

**Ready to clone your voice with Echo Cloner? Start by uploading audio files!**
