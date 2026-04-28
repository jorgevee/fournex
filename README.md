start backend w/: uvicorn api:app --reload
# GPU-RL — PyTorch & CUDA Performance Optimizer

An open-source CLI tool to help profile, analyze, and optimize PyTorch and CUDA performance.

## Features

- Bottleneck detection and classification
- Rule-based recommendations for GPU workloads
- Telemetry collection and trace analysis
- ROI ranking system for optimization suggestions
- REST API backend for integration with frontends or CI pipelines

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- Node.js 18+ (for frontend)

### Backend Setup

```bash
pip install -r requirements.txt
```

Start the backend API server:

```bash
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`.

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:3000`.

## Project Structure

```
semiconductor_eda/
├── backend/
│   ├── api.py                  # FastAPI application entry point
│   ├── python/                 # Core optimization logic
│   ├── schemas/                # Shared data schemas
│   ├── traces/                 # Trace capture utilities
│   ├── notebooks/              # Jupyter analysis notebooks
│   ├── tests/                  # Backend test suite
│   └── data/                   # Sample datasets
└── frontend/
    ├── app/                    # Next.js app directory
    └── public/                 # Static assets
```

## Usage

1. Start the backend server (`uvicorn api:app --reload`)
2. Run your PyTorch workload with telemetry enabled
3. View bottleneck analysis and optimization recommendations via the API or frontend UI

### CLI collector

Install the local CLI into the Python environment you use for test workloads:

```bash
cd backend/python
python -m pip install -e .
```

Then run a workload from any folder:

```bash
frx collect --name slow-input-test --out runs -- python train.py
```

For local development without installing the console script, point Python at the package:

```bash
PYTHONPATH=/c/Users/jorge/Documents/app_testing2/semiconductor_eda/backend/python \
python -m autopilot_telemetry collect --name slow-input-test --out runs -- python train.py
```

## Contributing

Pull requests are welcome. Please open an issue first to discuss proposed changes.

## License

MIT
