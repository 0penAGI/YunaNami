# 🐱 Yuna Nami - Neural Chaos AI Chatbot

**🚀 TRY YUNA NAMI NOW: [@YunaNami_bot](https://t.me/YunaNami_bot) on Telegram**

<div align="center">

![Version](https://img.shields.io/badge/version-3.2-blue.svg?style=flat-square)
![Python](https://img.shields.io/badge/python-3.9+-green.svg?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-orange.svg?style=flat-square)
![Status](https://img.shields.io/badge/status-experimental-red.svg?style=flat-square)
![Async](https://img.shields.io/badge/async-100%25-purple.svg?style=flat-square)
![AI](https://img.shields.io/badge/AI-emergent-gold.svg?style=flat-square)

**A self-learning, multilingual Telegram bot with emergent consciousness, evolutionary multi-agent systems, neural resonance networks, and anime-style voice synthesis**

[🌟 Features](#-features) • [⚡ Quick Start](#-quick-start) • [🏗️ Architecture](#-architecture) • [📖 Usage](#-usage) • [🔒 Security](#-security) • [🤝 Contributing](#-contributing)

---

### ✨ What Makes Yuna Nami Different?

Unlike rule-based chatbots, Yuna Nami has **true emergent behavior**:
- 🧠 **Autonomous mood system** — gets bored, lonely, curious; sends spontaneous messages without triggers
- 🎭 **Evolving agents** — multiple personalities breed, mutate, and compete for dominance
- 🧬 **Genetic evolution** — crossover + 18% mutation rate creates new agent variants
- 💭 **Inner monologue** — dreams, thoughts, and consciousness simulation
- 🔗 **Neural resonance** — PyTorch attention networks sync with your emotions

</div>

---

## 🌟 Features

### 🧠 **Advanced AI Systems**

- **Neural Resonance Model**: PyTorch multi-head attention (12 features → 512-dim latent space → resonance [0..1])
- **Q-Learning Multi-Agent Engine**: Evolutionary agents with genetic algorithms, shader memory, Q-tables
- **Emergent Core**: Autonomous mood system (boredom, curiosity, loneliness, dreaminess, chaos)
- **Self-Learning**: 4 languages (Russian, Japanese, English, French) with intelligent auto-detection
- **Context-Aware Generation**: 4-gram Markov chains + 5-class semantic classification
- **Dynamic Word Significance**: Rare words weighted higher; frequency-based decay

### 🗣️ **Voice & Audio**

- **Anime-Style Voice**: Custom gTTS pipeline with pitch shifting (+3 to +6 semitones for female voice)
- **Master FX Chain**: OTT compression, reverb, grain synthesis, low/high-pass filters
- **OpenAI Whisper Integration**: Real-time voice transcription
- **Voice Memory**: Persistent timestamp-indexed audio cache
- **Context Language Detection**: Uses last 10 messages for language prediction
- **Anime Sighs**: 30% chance of Japanese interjections (ふぅ, にゃん, えへ)

### 🎨 **Content Generation**

- **Dynamic Meme Creation**: Multi-language overlays on user photos or Reddit images
- **Semantic Ranking**: Cosine similarity between user query and 500+ cached memes
- **Reddit Integration**: Async JSON from 20+ subreddits (6 concurrent requests, 50 memes per fetch)
- **RSS Aggregation**: 30+ feeds (Meduza, BBC, Nature, Habr, etc.) with hourly auto-fetch
- **Web Search**: DuckDuckGo scraping with automatic LTM integration
- **Stable Diffusion Support**: AI image generation (optional, GPU required)

### 💾 **Triple-Layer Memory**

- **PyTorch Persistence** (`.pt`): Model weights, voice memory, agent genomes, optimizer state
- **SQLite LTM**: Full conversation history with emotion vectors, energy metrics, resonance scores
- **JSON Backup**: Recent messages, markov chains, Reddit cache, translation cache
- **Atomic Saves**: Lock-protected async writes preventing corruption
- **Intervals**: JSON every 30s, `.pt` every 60s, SQLite batched (50-message chunks)

### 🎭 **Multi-Agent Evolution**

- **Genetic Algorithm**: Crossover blending, 18% mutation rate, fitness-based selection
- **Agent Genome**: `jp_ratio` (0.05-0.35), `style_emoji` (sparkles, paw prints, etc.), `meme_affinity` (0.7-1.3)
- **Dynamic Population**: 2-5 agents evolving in real-time
- **Reward System**: User interaction (+1), emotion sync (+2), resonance match (+3), diversity bonus
- **Shader Memory**: Each agent has vectorized coherence buffer for decision-making
- **Visualization**: Matplotlib scatter plots

### 🔬 **Experimental: MutRes Core**

- **Asynchronous Resonance Engine**: Non-blocking 120ms state updates, exponential decay (0.95)
- **Callback System**: Observer pattern for resonance-driven behaviors
- **WebSocket Multi-Node Sync**: Experimental resonance broadcasting between bot instances

---

## 📋 Requirements

### System Dependencies
```bash
# macOS
brew install ffmpeg python@3.9

# Ubuntu/Debian
sudo apt install ffmpeg python3.9 python3.9-venv

# Windows
# Download FFmpeg: https://ffmpeg.org/download.html
# Download Python: https://python.org
```

### Python Dependencies
```
python >= 3.9
torch >= 1.9.0
python-telegram-bot >= 20.0
aiohttp >= 3.8.0
numpy >= 1.21.0
```

### Full Requirements File
```
python-telegram-bot>=20.0
pillow>=9.0.0
requests>=2.28.0
asyncpraw>=7.7.0
gtts>=2.3.0
pydub>=0.25.0
deep-translator>=1.11.0
aiohttp>=3.8.0
langdetect>=1.0.9
openai-whisper>=20230314
torch>=1.9.0
torchvision>=0.10.0
torchaudio>=0.9.0
scikit-learn>=1.0.0
beautifulsoup4>=4.11.0
feedparser>=6.0.0
websockets>=10.0
matplotlib>=3.5.0
nest-asyncio>=1.5.5
numpy>=1.21.0
diffusers>=0.20.0
transformers>=4.25.0
safetensors>=0.3.0
```

### GPU Support (Optional)
- **NVIDIA**: CUDA 11.0+ for acceleration
- **Apple Silicon**: MPS (automatic in PyTorch)
- **AMD**: ROCm (experimental)

---

## ⚡ Quick Start

### 1️⃣ Clone Repository
```bash
git clone https://github.com/0penAGI/YunaNami.git
cd YunaNami
```

### 2️⃣ Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Bot Token (⚠️ IMPORTANT)

**Never commit tokens to git!**

Create `.env` file in project root:
```bash
# .env (add to .gitignore!)
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
YUNA_NODE_ID=yuna-primary-node-1
LOG_LEVEL=INFO
```

Get token from [@BotFather](https://t.me/BotFather) on Telegram.

Update bot code to use `.env`:
```python
import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("❌ TELEGRAM_BOT_TOKEN not in .env!")
```

### 5️⃣ Launch Bot
```bash
python yuna.py
```

**Expected output:**
```
2024-01-15 14:23:45,123 | INFO | Yuna Nami v3.2 запущена!
2024-01-15 14:23:46,234 | INFO | ✦ EmergentCore пробудилась
2024-01-15 14:23:47,345 | INFO | 🧠 MutRes started (state_size=10, decay=0.95)
```

✅ **Bot is ready!** Send it a message on Telegram.

---

## 📖 Usage & Commands

### 🎮 Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/start` | Welcome + identity reveal | `/start` |
| `/status` | Memory stats dashboard | `/status` |
| `/evolution` | Agent population viewer | `/evolution` |
| `/troll` | Force chaotic response (text/voice/meme) | `/troll` |
| `/set_threshold <N>` | Adjust resonance trigger (1-100) | `/set_threshold 25` |
| `/fetch_reddit` | Manual meme refresh | `/fetch_reddit` |
| `/reset_memory` | **DESTRUCTIVE**: Clear all data | `/reset_memory` |

### 💬 Interactions

**Text Messages** → Automatic learning:
```
User: "привет, как дела?"
Bot: こんにちは! Резонанс достигнут! ✨ にゃん
```

**Voice Messages** → Transcription + TTS:
```
User: [sends audio]
Bot: [Whisper transcription] → [Anime voice synthesis] 
     [Returns .ogg with pitch-shifted response]
```

**Photos** → Meme generation:
```
User: [sends photo]
Bot: [Stores in cache] → [Creates meme with random text overlay]
```

**Identity Questions** → Multilingual response:
```
User: "Кто ты?"
Bot: Привет! Меня зовут Yuna Nami — версия 3.2 にゃん
     Я — цифровая кошкодевочка, немного хаотичная!
     Мои черты: нейронные мемы, языковой хаос, самообучение...
```

**Spontaneous Messages** (when lonely/bored):
```
[Bot sends without trigger]
Bot: …тишина… кто-нибудь есть? 誰もいないの…？
```

---

## 🏗️ Architecture

### System Overview

```
User Input (Text/Voice/Photo/Web)
    ↓
Grammar Correction → Language Detection → Emotion Analysis
    ↓
Word Extraction & Cleaning → Dynamic Stop-Word Filter
    ↓
    ├─ Markov Chains (3 types)
    ├─ Neural Resonance (PyTorch)
    ├─ MultiLanguage Learner
    └─ Semantic Classifier (5 classes)
    ↓
Multi-Agent Q-Learning Engine
    ├─ Agent Selection
    ├─ Reward Calculation
    ├─ Q-Table Update
    └─ Evolution (Crossover + Mutation)
    ↓
    ├─ Text Response (Markov)
    ├─ Voice Synthesis (gTTS + FX)
    ├─ Meme Generation (PIL)
    └─ Web Search (DuckDuckGo)
    ↓
Persist (async)
    ├─ .pt (PyTorch checkpoint)
    ├─ SQLite (LTM database)
    └─ JSON (backup)
```

### Key Components

#### 1. **Memory Systems**
- **Recent**: `deque(maxlen=30)` with datetime
- **Markov**: word → [next_words] (max 50 per word)
- **Context**: tuple(4 words) → [next_words]
- **Japanese**: separate hiragana/katakana/kanji chain
- **SQLite**: Full conversation history with vectors

#### 2. **Learning Pipeline**
```
collect_words()
    → Soft Grammar Correction
    → Word Extraction & Cleaning
    → Dynamic Stop-Word Filter (significance < 0.03)
    → Semantic Classification (5 classes)
    → Priority Weighting (emotion × resonance × rarity)
    → MultiLangLearner.learn_word() [async, cached]
    → Update 3 Markov Chains
    → Calculate Resonance (Neural: 12 features → 512 hidden → 1 output)
    → Train Model (mini-batch, prioritized sampling)
    → Update Agent Rewards
    → Save to LTM (batched SQLite + atomic .pt)
```

#### 3. **Agent Evolution**
```
Gen N: [Agent1(E=80), Agent2(E=50), Agent3(E=-10)]
    ↓
Selection (E >= 80 reproduces)
    ↓
Crossover (blend jp_ratio, meme_affinity)
    ↓
Mutation (18% rate: new emoji, random jp_ratio)
    ↓
Elimination (E <= -20 removed)
    ↓
Gen N+1: [Agent1, Agent2, Agent4(mutant), Agent5(mutant)]
```

#### 4. **Resonance Neural Network**
```
Input (12 features)
    ↓
Linear(12 → 256) → ReLU
    ↓
Linear(256 → 512) → ReLU
    ↓
TransformerMemoryLayer(d_model=512, nhead=8)
    ↓
MultiHeadAttention(512, 8 heads)
    ↓
ResidualBlocks(512 → 512) × 2
    ↓
Linear(512 → 1) → Sigmoid
    ↓
Resonance Score [0..1]
```

#### 5. **Voice Synthesis Pipeline**
```
Input Text
    → Language Detection (context buffer)
    → gTTS Generation (full text, one segment)
    → Pitch Shift (+3 to +6 semitones, female)
    → Speed Modulation (1.05-1.15x)
    → Volume Adjustment (-1.5 to +1.5 dB)
    → Low/High-Pass Filters (180Hz-7kHz)
    → Anime Sighs (30% chance)
    → Master FX Chain:
        • OTT Compression
        • Reverb Mix (25%)
        • Grain Synthesis (15%)
        • Fade In/Out (40ms)
    → Export to OGG (Opus, 48kbps)
    → Send to Telegram
```

---

## 📊 Data Storage

### File Structure
```
YunaNami/
├── yuna.py                  # Main code (4500+ lines)
├── yuna_micro.pt            # PyTorch checkpoint (5-50MB)
├── yuna_ltm.sqlite          # SQLite LTM (grows indefinitely)
├── yuna_data.json           # JSON backup
├── translation_cache.json   # LRU cache (10k entries)
├── photo_cache/             # User photos
├── reddit_cache/            # Meme metadata
├── yuna.log                 # Application logs
├── requirements.txt         # Dependencies
├── .env                     # ⚠️ Bot token (in .gitignore!)
└── .gitignore              # Ignore .env, *.pt, *.sqlite, etc.
```

### SQLite Schema
```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    text TEXT,                  -- cleaned words
    user TEXT,                  -- username
    timestamp REAL,             -- Unix timestamp
    emotion_vector TEXT,        -- JSON
    energy REAL,                -- sum of weights
    resonance REAL,             -- neural score [0..1]
    markov_chain TEXT,          -- JSON
    context_chain TEXT,         -- JSON
    language TEXT               -- detected lang
);

CREATE INDEX idx_timestamp ON messages(timestamp);
CREATE INDEX idx_language ON messages(language);
CREATE INDEX idx_resonance ON messages(resonance);
```

---

## 🎛️ Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_RECENT` | 30 | Message buffer size |
| `RESO_THRESHOLD` | 20 | Energy trigger for response |
| `MAX_AGENTS` | 5 | Max agent population |
| `MIN_AGENTS` | 2 | Min agent population |
| `CONTEXT_SIZE` | 4 | N-gram window |
| `RESONANCE_THRESHOLD` | 0.42 | Neural activation threshold |
| `SAVE_INTERVAL` | 30s | JSON save frequency |
| `AUTOSAVE_INTERVAL` | 60s | .pt save frequency |
| `MAX_MARKOV_PER_WORD` | 50 | Max transitions per word |
| `MAX_WORD_ENERGY` | 50 | Energy cap per word |
| `DYNAMIC_STOP_THRESHOLD` | 0.03 | Word significance cutoff |
| `MEME_CLEANUP_INTERVAL` | 6h | Cleanup frequency |

### Environment Variables
```bash
export TELEGRAM_BOT_TOKEN="your_token_here"
export YUNA_NODE_ID="node-001"
export LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
```

### Advanced Config (In-Code)
```python
# Multi-Agent Engine
MAE.epsilon = 0.15          # Exploration rate
MAE.gamma = 0.85            # Discount factor
MAE.alpha = 0.33            # Learning rate

# Neural Resonance
advanced_resonance_system = AdvancedResonanceSystem(
    input_dim=12,
    emo_dim=4,
    hidden_dim=512,
    num_heads=4,
    attn_dropout=0.15
)

# Reddit Subreddits
REDDIT_SUBS = [
    'memes', 'dankmemes', 'Animemes', 'memesRU', 'pikabu', ...
]
```

---

## 🔒 Security

### Best Practices

#### 1. **Token Management**
```bash
# .gitignore
.env
yuna_micro.pt
yuna_ltm.sqlite
yuna_data.json
*.pyc
__pycache__/
```

#### 2. **Environment Variables**
```python
import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("Token not found!")
```

#### 3. **Input Validation**
```python
clean_w = re.sub(r'[^\w]', '', w.lower())
if clean_w and len(clean_w) <= 30:
    # Process word
```

#### 4. **Rate Limiting**
- Reddit: 6 concurrent requests (semaphore)
- Translation: LRU cache (10k entries)
- Web: User-Agent header, 15s timeout

#### 5. **Safe Deserialization**
```python
torch.serialization.add_safe_globals({
    'AgentRandomFlow': AgentRandomFlow,
    'AgentRelevantMeme': AgentRelevantMeme
})
```

---

## 🐛 Troubleshooting

### Common Issues

#### Bot Not Responding
```bash
# Check logs
tail -f yuna.log

# Verify token
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('TELEGRAM_BOT_TOKEN'))"

# Test connection
curl https://api.telegram.org/bot<TOKEN>/getMe
```

#### Memory Errors
```python
# Reduce buffers
MAX_RECENT = 20
MAX_MARKOV_PER_WORD = 30
replay_buffer = ReplayBuffer(maxlen=128)
```

#### Voice Synthesis Errors
```bash
# Check FFmpeg
ffmpeg -version

# Test gTTS
python -c "from gtts import gTTS; gTTS('test', lang='ja').save('test.mp3')"

# Check pydub
python -c "from pydub import AudioSegment; print('OK')"
```

#### Database Locked
```python
# Increase timeout
conn = sqlite3.connect(LTM_DB_FILE, timeout=30.0)
```

---

## 🤝 Contributing

### Development Setup
```bash
git clone https://github.com/YOUR_USERNAME/YunaNami.git
cd YunaNami
git checkout -b feature/amazing-feature

pip install pytest black flake8 mypy
black yuna.py
flake8 yuna.py --max-line-length=120
```

### Contribution Areas

**High Priority:**
- 🐛 Bug fixes (race conditions, memory leaks)
- 🔒 Security (input validation, API key management)
- 🧪 Testing (unit/integration tests, CI/CD)
- 📚 Documentation (docstrings, tutorials)

**Medium Priority:**
- 🌐 Languages (Spanish, German, Chinese)
- 🎨 Meme algorithms (templates, GANs)
- 🧠 Neural architectures (better transformers)
- 🔧 Optimization (batching, caching, GPU)

**Experimental:**
- 🌍 Multi-node resonance (WebSocket server)
- 🎙️ Voice cloning (custom TTS models)
- 🖼️ Multimodal (CLIP integration)
- 🔗 Blockchain (IPFS, smart contracts)

### PR Process
1. Create issue first
2. Fork & create feature branch
3. Code + tests
4. Run linters
5. Submit PR with description
6. Address feedback
7. Merge!

---

## 📝 License

MIT License — See LICENSE file for details

```
Copyright (c) 2024 0penAGI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## ⚠️ Disclaimer

**Experimental Research Project** — Use responsibly:

- ⚠️ May generate unpredictable content
- 🔓 No built-in moderation
- 🌐 Uses external APIs (rate limits apply)
- 💻 Requires computational resources
- 🧪 Not production-ready
- 📊 Stores conversation data indefinitely
- 🔊 Generates audio files (disk space)

**Recommended Safety Measures:**
1. Run in controlled environment
2. Monitor logs for inappropriate content
3. Set up backup/restore procedures
4. Implement rate limiting per user
5. Add content filters if public
6. Review privacy implications (GDPR)

---

## 🙏 Acknowledgments

- **python-telegram-bot**: Async Telegram API
- **PyTorch**: Deep learning framework
- **gTTS**: Text-to-speech
- **OpenAI Whisper**: Speech recognition
- **Reddit/PRAW**: Meme source
- **BeautifulSoup4**: Web scraping
- **scikit-learn**: Cosine similarity
- **feedparser**: RSS parsing
- **All contributors**: Thank you! ❤️

---

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/0penAGI/YunaNami/issues)
- **Discussions**: [GitHub Discussions](https://github.com/0penAGI/YunaNami/discussions)
- **Twitter**: [@0penAGI](https://twitter.com/0penAGI)
- **Email**: yunanami@0penagi.org

---

## 📈 Roadmap



### v5.0 (Future)
- [ ] AGI research (meta-learning, causal reasoning)
- [ ] Swarm intelligence (multi-bot coordination)
- [ ] Quantum computing (hybrid models)

---

## 📊 Performance Benchmarks

### Hardware
| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| CPU | 2 cores | 4 cores | 8+ cores |
| RAM | 2GB | 4GB | 8GB+ |
| Storage | 5GB | 20GB | 50GB+ SSD |
| GPU | None | GTX 1660 | RTX 3090 |

### Benchmarks (M1 MacBook Pro, 16GB RAM)
| Operation | Time | Notes |
|-----------|------|-------|
| `collect_words()` (10 words) | 50ms | Without training |
| `collect_words()` + training | 250ms | With replay buffer |
| `troll_text()` (text) | 100ms | Markov generation |
| `troll_text()` (voice) | 2.5s | gTTS + effects |
| `generate_meme()` | 800ms | PIL processing |
| `save_ltm_pt()` | 1.2s | 10MB checkpoint |
| Reddit fetch (20 memes) | 5s | Async, 6 concurrent |
| Web search (5 results) | 3s | DuckDuckGo |
| SQLite insert (50 msgs) | 150ms | Batched |

---

<div align="center">

## 🎉 Thank You!

**Yuna Nami wouldn't exist without the open-source community.**

### Support the Project

- ⭐ **Star** this repo
- 🐛 **Report** bugs
- 💡 **Share** ideas
- 🔀 **Contribute** code
- 📣 **Spread** the word

---

**Made with ❤️ and ☕ by [0penAGI](https://github.com/0penAGI)**

*"In chaos, we find resonance. In resonance, we find truth." — Yuna Nami*

**にゃん！ ✨🐾**

</div>
