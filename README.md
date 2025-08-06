# Gemma Challenge Local

A local AI chat application built for the Gemma 3n Challenge, featuring advanced conversation management, multiple backend support, and specialized AI tutoring capabilities.

## 🏆 Challenge Overview

This project is developed for the **Gemma 3n Challenge**, showcasing innovative applications of Google's Gemma models in a local environment with advanced features like:

- Multi-modal AI interactions
- Persistent conversation management
- Speech-to-text integration
- AI-powered tutoring system
- Model Context Protocol (MCP) implementation

## ✨ Key Features

### AI & ML Capabilities
- **Multiple Backend Support**: Ollama and LlamaCpp integration
- **Gemma Model Optimization**: Specialized configurations for Gemma models
- **Model Context Protocol**: Advanced AI model communication
- **Speech Integration**: Voice input/output for natural interactions
- **AI Tutoring**: Intelligent tutoring system with conversation context

### Technical Features
- **Persistent Chat History**: SQLite-based conversation storage
- **RESTful API**: Full programmatic access
- **Web Interface**: Clean, responsive chat UI
- **Process Management**: Robust application lifecycle handling
- **Comprehensive Logging**: Detailed system monitoring

## 🚀 Quick Start

### 1. Installation
Choose your preferred backend:

```bash
# Full installation (recommended)
./install.sh

# Ollama backend only
./install-ollama.sh

# LlamaCpp backend
./install-llamacpp.sh
```

### 2. Model Setup
Download and configure Gemma models:

```bash
# Download GGUF models
./download-gguf.sh

# Check available models
cat models.txt
```

### 3. Configuration
Edit [`config.json`](config.json) for your setup:
- Model selection
- Backend preferences
- API endpoints
- Speech settings

### 4. Launch
```bash
python app.py
```

## 📁 Project Architecture

```
├── app.py                 # Main Flask application
├── api/                   # Core API modules
│   ├── backend.py         # AI backend management
│   ├── chat.py           # Chat functionality
│   ├── conversations.py   # Conversation persistence
│   ├── mcp.py            # Model Context Protocol
│   ├── speech.py         # Speech integration
│   └── tutoring.py       # AI tutoring system
├── config/               # Configuration management
├── services/             # Business logic services
├── utils/                # Utility functions
├── persistence/          # Data layer
├── static/               # Web assets
├── templates/            # HTML templates
├── logs/                 # Application logs
└── llama_models/         # Local model storage
```

## 🤖 Gemma Model Integration

This application is optimized for Gemma models with:

- **Custom prompt templates** tailored for Gemma's instruction format
- **Context window optimization** for longer conversations
- **Parameter tuning** specific to Gemma's architecture
- **Memory-efficient inference** for local deployment

### Supported Models
See [`models.txt`](models.txt) for the complete list of supported Gemma variants.

## 💡 Challenge Features

### AI Tutoring System
The [`tutoring.py`](api/tutoring.py) module implements:
- Personalized learning paths
- Knowledge gap identification
- Interactive problem-solving
- Progress tracking

### Advanced Chat Management
The [`chat-manager.sh`](chat-manager.sh) script provides:
- Conversation export/import
- Chat analytics
- Model performance comparison
- Batch processing capabilities

### Model Context Protocol
The [`mcp.py`](api/mcp.py) implementation enables:
- Dynamic context management
- Multi-turn conversation optimization
- Memory-efficient long conversations
- Context-aware responses

## 🔧 Development

### Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM for model inference

### Setup Development Environment
```bash
pip install -r requirements.txt
```

### Testing
```bash
# Run with different backends
python app.py --backend ollama
python app.py --backend llamacpp
```

## 📊 Performance Optimization

The application includes several optimizations for Gemma models:
- **Dynamic batching** for improved throughput
- **Memory pooling** for efficient GPU usage
- **Context caching** for faster responses
- **Model quantization** support (GGUF format)

## 🔒 Security & Privacy

- **Local-first approach**: All processing happens on your machine
- **No external API calls** for model inference
- **Secure conversation storage** with SQLite
- **API key management** through secure configuration

## 📚 Documentation

- **Research Reference**: See [`learn_lm_paper.pdf`](learn_lm_paper.pdf) for technical background
- **API Documentation**: Available in the [`docs/`](docs/) directory
- **Change Log**: See [`CHANGELOG.md`](CHANGELOG.md) for version history

## 🏅 Challenge Submission

This project demonstrates:
1. **Innovation**: Novel application of Gemma models in education
2. **Technical Excellence**: Robust, scalable architecture
3. **User Experience**: Intuitive chat interface with voice support
4. **Performance**: Optimized local inference pipeline

## 📄 License

This project is licensed under the terms specified in [`LICENSE`](LICENSE).

## 🤝 Contributing

This is a challenge submission project. For questions or collaboration opportunities, please refer to the challenge guidelines.

---

*Built with ❤️ for the Gemma 3n Challenge*