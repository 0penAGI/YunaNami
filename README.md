# 🐱 Yuna Nami - Neural Chaos AI Chatbot

<div align="center">

![Version](https://img.shields.io/badge/version-3.2-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)
![Status](https://img.shields.io/badge/status-experimental-red.svg)

**A self-learning, multilingual Telegram bot with evolutionary multi-agent systems, neural resonance, and anime-style voice synthesis**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Contributing](#-contributing)

</div>

---

## 🌟 Features

### 🧠 **Advanced AI Systems**
- **Neural Resonance Model**: PyTorch-based deep learning with attention mechanisms
- **Q-Learning Multi-Agent Engine**: Evolutionary agents that adapt and evolve through natural selection
- **Self-Learning**: Automatic vocabulary expansion across 4 languages (Russian, Japanese, English, French)
- **Context-Aware Generation**: N-gram Markov chains with 4-word context windows

### 🗣️ **Voice Synthesis**
- **Anime-Style Voice**: Custom gTTS pipeline with pitch shifting, speed modulation, and FX processing
- **Multilingual Support**: Automatic language detection and voice synthesis
- **Voice Memory**: Persistent audio cache for pattern analysis

### 🎨 **Content Generation**
- **Dynamic Meme Creation**: Text overlay on user photos or Reddit images
- **Reddit Integration**: Async scraping from 20+ subreddits (memes, anime, programming)
- **RSS Feed Aggregation**: News, science, tech, and quotes from 30+ sources
- **Web Search**: DuckDuckGo integration with automatic knowledge acquisition

### 💾 **Persistent Memory**
- **Triple-Layer Storage**: PyTorch (.pt), SQLite database, and JSON backup
- **Long-Term Memory (LTM)**: Conversation history with emotion vectors and energy metrics
- **Atomic Saves**: Lock-protected data persistence preventing corruption

### 🎭 **Multi-Agent Evolution**
- **Genetic Algorithm**: Crossover, mutation, and natural selection
- **Agent Genome**: jp_ratio, style_emoji, meme_affinity genes
- **Dynamic Population**: 2-5 agents, evolving based on performance
- **Reward System**: User interaction, emotion matching, resonance alignment

---

## 📋 Requirements

### Core Dependencies
```bash
python >= 3.8
torch >= 1.9.0
python-telegram-bot >= 20.0
asyncio
aiohttp
```

### Full Dependencies
```bash
# Install all requirements
pip install python-telegram-bot pillow requests asyncpraw gtts pydub \
            libretranslatepy deep-translator aiohttp langdetect openai-whisper \
            torch numpy scikit-learn beautifulsoup4 feedparser nest-asyncio \
            matplotlib websockets
```

### System Requirements
- **FFmpeg**: Required for audio processing
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/)

---

## 🚀 Installation

### 1. Clone Repository
```bash
git clone https://github.com/0penAGI/YunaNami.git
cd YunaNami
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Bot Token
Edit `yuma.py` and replace the bot token:
```python
# Line ~4500
app = Application.builder().token("YOUR_BOT_TOKEN_HERE").build()
```

Get your token from [@BotFather](https://t.me/BotFather) on Telegram.

### 5. Run the Bot
```bash
python yuma.py
```

---

## 💻 Usage

### Basic Commands

| Command | Description |
|---------|-------------|
| `/start` | Initialize bot and show welcome message |
| `/status` | View memory stats, word count, energy levels |
| `/evolution` | Display agent population and fitness |
| `/troll` | Manually trigger chaotic response |
| `/reset_memory` | Clear all learned data (destructive!) |
| `/set_threshold <N>` | Adjust resonance trigger threshold |
| `/fetch_reddit` | Manually refresh Reddit meme cache |

### Interaction Examples

**Text Learning:**
```
User: "Hello, how are you?"
Bot: こんにちは! I'm in flow state ✨
```

**Voice Interaction:**
- Send voice message → Bot transcribes with Whisper
- Bot responds with anime-style synthesized voice
- Automatic language detection (Japanese/Russian/English/French)

**Meme Generation:**
- Send photos → Bot learns from images
- Automatic meme creation when energy threshold reached
- Multi-language text overlays with chaos effects

---

## 🏗️ Architecture

### System Overview
```
┌─────────────────────────────────────────────────────────────┐
│                         User Input                          │
│              (Text, Voice, Photos, Commands)                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Message Processor                         │
│  • Grammar correction  • Word extraction  • Language detect │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌────────┐    ┌──────────┐    ┌──────────┐
    │Markov  │    │Resonance │    │Multi-    │
    │Chains  │    │Neural Net│    │Language  │
    └────────┘    └──────────┘    │Learner   │
         │               │         └──────────┘
         │               ▼               │
         │        ┌──────────┐          │
         └───────►│Multi-Agent│◄─────────┘
                  │Q-Learning │
                  └──────┬─────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌────────┐    ┌──────────┐    ┌──────────┐
    │  Text  │    │  Voice   │    │   Meme   │
    │Response│    │Synthesis │    │Generator │
    └────────┘    └──────────┘    └──────────┘
         │               │               │
         └───────────────┼───────────────┘
                         ▼
                 ┌────────────┐
                 │   Output   │
                 └────────────┘
                         │
                         ▼
                 ┌────────────┐
                 │  Persist   │
                 │ .pt / SQL  │
                 └────────────┘
```

### Key Components

#### 1. **Memory Systems**
- **Recent Messages**: Deque (30 messages)
- **Markov Chains**: Word transitions with context
- **SQLite LTM**: Full conversation history
- **PyTorch State**: Neural model weights

#### 2. **Learning Pipeline**
```python
collect_words() → MultiLangLearner.learn_word()
                → update_markov_chain()
                → calculate_resonance()
                → train_neural_model()
                → save_to_ltm()
```

#### 3. **Agent Evolution**
```
Generation N:
[Agent1(E=80), Agent2(E=50), Agent3(E=-10)]
                    ↓
         Selection & Crossover
                    ↓
Generation N+1:
[Agent1, Agent2, Agent4(mutant), Agent5(mutant)]
```

#### 4. **Resonance Calculation**
```python
Features → [lang_sync, emotion_sync, semantic_sync, 
           joy, tension, flow, surprise,
           energy, word_count, time_of_day]
           ↓
    Neural Network (10→24→1)
           ↓
    Resonance Score [0..1]
```

---

## 📊 Data Storage

### File Structure
```
YunaNami/
├── yuna.py              # Main bot code
├── yuna_micro.pt        # PyTorch model & memory
├── yuna_ltm.sqlite      # SQLite conversation DB
├── yuna_data.json       # JSON backup
├── photo_cache/         # User photo storage
├── reddit_cache/        # Reddit meme cache
└── yuma.log            # Application logs
```

### Database Schema (SQLite)
```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    text TEXT,
    user TEXT,
    timestamp REAL,
    emotion_vector TEXT,
    energy REAL,
    resonance REAL,
    markov_chain TEXT,
    context_chain TEXT,
    language TEXT
);
```

---

## 🎛️ Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_RECENT` | 30 | Recent message buffer size |
| `RESO_THRESHOLD` | 20 | Energy trigger for response |
| `MAX_AGENTS` | 5 | Maximum agent population |
| `CONTEXT_SIZE` | 4 | N-gram context window |
| `RESONANCE_THRESHOLD` | 0.42 | Neural activation threshold |
| `SAVE_INTERVAL` | 30s | Auto-save frequency |

### Environment Variables
```bash
export YUNA_NODE_ID="unique-node-identifier"  # For multi-node setups
```

---

## 🔬 Experimental Features

### MutRes Core
Asynchronous resonance engine with:
- Non-blocking state updates (120ms intervals)
- Callback system for observers
- Exponential decay (0.95 factor)

### Resonance Synchronization Protocol
**WebSocket-based multi-node network** (experimental):
```python
# Enable multi-node resonance sync
asyncio.create_task(resonance_sync_loop("ws://hub.example.com:8765"))
```

Broadcasts resonance packets across nodes for emergent behavior.

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Development Setup
```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/YunaNami.git

# Create feature branch
git checkout -b feature/amazing-feature

# Install dev dependencies
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black yuna.py
```

### Contribution Areas
- 🐛 Bug fixes and stability improvements
- 🌐 Additional language support
- 🎨 New meme generation algorithms
- 🧠 Enhanced neural architectures
- 📚 Documentation and examples
- 🧪 Unit tests and integration tests

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for public functions
- Keep functions under 50 lines when possible

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 0penAGI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

---

## ⚠️ Disclaimer

**This is an experimental research project.** The bot:
- May generate unpredictable content
- Learns from user input (review conversations)
- Uses external APIs (Reddit, RSS, translation services)
- Requires computational resources (neural model training)
- Is not suitable for production without additional safety measures

**Use responsibly and at your own risk.**

---

## 🙏 Acknowledgments

- **python-telegram-bot**: Excellent async Telegram API wrapper
- **PyTorch**: Deep learning framework
- **gTTS**: Text-to-speech synthesis
- **OpenAI Whisper**: Speech recognition
- **Reddit API**: Meme content source
- **Contributors**: Thanks to all who have contributed!

---

## 📧 Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/0penAGI/YunaNami/issues)
- **Discussions**: [Join the conversation](https://github.com/0penAGI/YunaNami/discussions)
- **Twitter**: [@0penAGI](https://twitter.com/0penAGI)


---

## 📈 Roadmap

### v3.3 (Planned)
- [ ] Stable Diffusion integration for AI meme generation
- [ ] Advanced grammar correction with context
- [ ] Voice cloning for personalized synthesis
- [ ] Multi-modal training (text + image + audio)
- [ ] Distributed resonance network (production-ready)

### v4.0 (Future)
- [ ] GPT integration for natural language understanding
- [ ] Real-time collaboration features
- [ ] Mobile app companion
- [ ] Custom agent designer UI
- [ ] Blockchain-based memory persistence

---

<div align="center">

**Made with ❤️ by [0penAGI](https://github.com/0penAGI)**

⭐ Star this repo if you find it interesting!

</div>
