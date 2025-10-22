# LLM Model Converter for Ollama

A modular GUI application built with CustomTkinter to download models from Hugging Face and CivitAI, convert them to Ollama-compatible formats, and manage local model libraries.

## Features
- Modern theming with CustomTkinter
- Configuration management with persistent storage
- Logging to both console and rotating file handler
- Tabbed interface prepared for download, conversion, and library management workflows

## Requirements
- Python 3.8+
- Ollama installed and accessible in your PATH

## Setup
1. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage
Run the application entry point:
```powershell
python -m src.main
```

## Project Structure
```
.
├── downloads/
├── logs/
├── requirements.txt
├── README.md
├── src/
│   ├── core/
│   ├── gui/
│   └── utils/
```

## License
This project is released under the MIT License.
