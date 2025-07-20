# Ulama LLM Trainer

A web-based interface for fine-tuning LLM models with custom knowledge using Ollama. This application allows you to upload training data, configure model parameters, and train custom models with ease.

## Features

- Upload training data in various formats (TXT, JSON, JSONL)
- Select from available Ollama base models
- Customize training parameters (learning rate, batch size, epochs)
- Add custom system prompts
- Monitor training progress in real-time
- Simple and intuitive user interface

## Prerequisites

- Python 3.8+
- Node.js 16+
- Ollama installed and running locally

## Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\\venv\\Scripts\\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the backend server:
   ```bash
   uvicorn main:app --reload
   ```

The backend will be available at `http://localhost:8000`

## Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

The frontend will be available at `http://localhost:3000`

## Usage

1. Open the application in your web browser
2. Upload one or more training files
3. Select a base model from the dropdown
4. (Optional) Add a custom system prompt
5. Click "Start Training" to begin the fine-tuning process
6. Monitor the training progress in the progress bar

## Project Structure

```
├── backend/
│   ├── main.py           # FastAPI application
│   ├── requirements.txt   # Python dependencies
│   ├── uploads/          # Temporary storage for uploaded files
│   └── models/           # Trained models will be saved here
├── frontend/
│   ├── pages/            # Next.js pages
│   ├── public/           # Static files
│   ├── styles/           # Global styles
│   ├── package.json      # Frontend dependencies
│   └── tailwind.config.js # Tailwind CSS configuration
└── README.md             # This file
```

## License

This project is open source and available under the [MIT License](LICENSE).
