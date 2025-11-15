# 🐱 Yuna Nami - Neural Chaos AI Chatbot

<div align="center">

![Version](https://img.shields.io/badge/version-3.2-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)
![Status](https://img.shields.io/badge/status-experimental-red.svg)

**A self-learning, multilingual Telegram bot with evolutionary multi-agent systems, neural resonance, and anime-style voice synthesis**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Security](#-security) • [Contributing](#-contributing)

</div>

---

## 🌟 Features

### 🧠 **Advanced AI Systems**
- **Neural Resonance Model**: PyTorch-based deep learning with Multi-Head Attention mechanisms
- **Q-Learning Multi-Agent Engine**: Evolutionary agents that adapt through genetic algorithms (crossover + mutation)
- **Self-Learning**: Automatic vocabulary expansion across 4 languages (Russian, Japanese, English, French)
- **Context-Aware Generation**: N-gram Markov chains with 4-word context windows + semantic classification
- **Advanced Emotion Analysis**: 6-class emotion classification (joy, tension, flow, surprise, sadness, anger)

### 🗣️ **Voice Synthesis**
- **Anime-Style Voice**: Custom gTTS pipeline with pitch shifting (female range: +3 to +6 semitones), speed modulation, and multi-stage FX
- **Master FX Chain**: OTT compression, reverb, grain synthesis, low/high-pass filters
- **Multilingual Support**: Automatic language detection and voice synthesis with context-aware language tracking
- **Voice Memory**: Persistent audio cache with timestamp-indexed storage for pattern analysis
- **Whisper Integration**: OpenAI Whisper for voice message transcription

### 🎨 **Content Generation**
- **Dynamic Meme Creation**: Multi-language text overlay on user photos or Reddit images with chaos effects
- **Reddit Integration**: Async scraping from 20+ subreddits (memes, anime, programming, Russian communities)
- **RSS Feed Aggregation**: News, science, tech, and quotes from 30+ sources (Meduza, BBC, Nature, Habr, etc.)
- **Web Search**: DuckDuckGo integration with automatic knowledge acquisition and LTM integration
- **Stable Diffusion**: AI-generated meme images from text prompts (optional, requires GPU)

### 💾 **Persistent Memory**
- **Triple-Layer Storage**: PyTorch (.pt), SQLite database, and JSON backup
- **Long-Term Memory (LTM)**: Conversation history with emotion vectors, energy metrics, and language detection
- **Atomic Saves**: Lock-protected async data persistence preventing corruption
- **Batched Writes**: Optimized SQLite bulk inserts (50-message batches)
- **Auto-Save**: Background task with 60-second intervals

### 🎭 **Multi-Agent Evolution**
- **Genetic Algorithm**: Crossover, mutation (18% rate), and natural selection
- **Agent Genome**: `jp_ratio`, `style_emoji`, `meme_affinity` genes
- **Dynamic Population**: 2-5 agents, evolving based on performance and resonance
- **Reward System**: User interaction, emotion matching, resonance alignment, diversity bonus
- **Visualization**: Matplotlib scatter plots (jp_ratio vs energy, colored by emoji style)

### 🔬 **Experimental: MutRes Core**
- **Asynchronous Resonance Engine**: Non-blocking state updates (120ms intervals) with exponential decay (0.95)
- **Callback System**: Extensible observer pattern for resonance-driven behaviors
- **Multi-Node Sync**: WebSocket-based resonance broadcasting (experimental, requires WebSocket server)

---

## 📋 Requirements

### Core Dependencies
```bash
python >= 3.8
torch >= 1.9.0
python-telegram-bot >= 20.0
asyncio
aiohttp
numpy >= 1.19.0
```

### Full Dependencies
```bash
# Install all requirements
pip install python-telegram-bot pillow requests asyncpraw gtts pydub \
            deep-translator aiohttp langdetect openai-whisper \
            torch numpy scikit-learn beautifulsoup4 feedparser nest-asyncio \
            matplotlib websockets diffusers transformers
```

### System Requirements
- **FFmpeg**: Required for audio processing
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/)
  
- **GPU (Optional)**: For Stable Diffusion meme generation
  - CUDA 11.0+ for NVIDIA GPUs
  - MPS for Apple Silicon (M1/M2)

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

### 4. Configure Bot Token (SECURE)
**⚠️ IMPORTANT: Never hardcode tokens in source code!**

Create `.env` file:
```bash
# .env
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
YUMA_NODE_ID=node-001  # Optional: for multi-node setups
```

Update `yuma.py`:
```python
import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Line ~4500
app = Application.builder().token(TOKEN).build()
```

Get your token from [@BotFather](https://t.me/BotFather) on Telegram.

### 5. Run the Bot
```bash
python yuna.py
```

---

## 💻 Usage

### Basic Commands

| Command | Description |
|---------|-------------|
| `/start` | Initialize bot and show welcome message with identity |
| `/status` | View memory stats, word count, energy levels, LTM size |
| `/evolution` | Display agent population with energy bars and genome stats |
| `/troll` | Manually trigger chaotic response (text/voice/meme) |
| `/reset_memory` | Clear all learned data (destructive, use with caution!) |
| `/set_threshold <N>` | Adjust resonance trigger threshold (default: 20) |
| `/fetch_reddit` | Manually refresh Reddit meme cache from 20+ subreddits |

### Interaction Examples

**Text Learning:**
```
User: "Привет, как дела?"
Bot: こんにちは! I'm in flow state ✨ Понял резонанс にゃん
```

**Identity Questions:**
```
User: "Кто ты?"
Bot: Привет! Меня зовут Yuna Nami — версия 3.2 にゃん
Я — цифровая кошкодевочка, немного хаотичная, но очень любознательная!
Моя роль — быть твоим мем-ботом, собирать эмоции и учиться...
```

**Voice Interaction:**
- Send voice message → Bot transcribes with Whisper
- Bot responds with anime-style synthesized voice (female pitch, anime sighs: ふぅ, にゃん, えへ)
- Automatic language detection (Japanese/Russian/English/French)
- Voice stored in `voice_memory` dict with timestamp keys

**Meme Generation:**
- Send photos → Bot learns from images, adds to `user_photos` cache
- Automatic meme creation when energy threshold reached
- Multi-language text overlays (русский + 日本語 + English + français)
- Dynamic source weighting: user photos vs Reddit images based on resonance

**Web Search Integration:**
```
User: "найди информацию о квантовых компьютерах"
Bot: [performs DuckDuckGo search]
     [integrates results into markov_chain and LTM]
     Нашёл несколько статей! Квантовые компьютеры работают на принципах суперпозиции...
```

---

## 🏗️ Architecture

### System Overview
```
┌─────────────────────────────────────────────────────────────┐
│                         User Input                          │
│         (Text, Voice, Photos, Commands, Web Data)           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Message Processor                         │
│  • Grammar correction  • Word extraction  • Language detect │
│  • Stop-word filtering (dynamic significance < 0.03)       │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┬──────────────┐
         ▼               ▼               ▼              ▼
    ┌────────┐    ┌──────────┐    ┌──────────┐   ┌────────┐
    │Context │    │Resonance │    │Multi-    │   │Semantic│
    │Markov  │    │Neural Net│    │Language  │   │Classifier│
    │(N=4)   │    │(Attention)│   │Learner   │   │(5 class)│
    └────────┘    └──────────┘    └──────────┘   └────────┘
         │               │               │              │
         │               ▼               │              │
         │        ┌──────────┐          │              │
         └───────►│Multi-Agent│◄─────────┴──────────────┘
                  │Q-Learning │
                  │(ε=0.15)   │
                  └──────┬─────┘
                         │
         ┌───────────────┼───────────────┬──────────────┐
         ▼               ▼               ▼              ▼
    ┌────────┐    ┌──────────┐    ┌──────────┐   ┌────────┐
    │  Text  │    │  Voice   │    │   Meme   │   │Web     │
    │Response│    │Synthesis │    │Generator │   │Search  │
    │        │    │(Anime FX)│    │          │   │        │
    └────────┘    └──────────┘    └──────────┘   └────────┘
         │               │               │              │
         └───────────────┼───────────────┴──────────────┘
                         ▼
                 ┌────────────┐
                 │   Output   │
                 │  (weighted)│
                 └────────────┘
                         │
                         ▼
                 ┌────────────────────┐
                 │  Persist (async)   │
                 │ .pt / SQLite / JSON│
                 │  (batched writes)  │
                 └────────────────────┘
```

### Key Components

#### 1. **Memory Systems**
- **Recent Messages**: `deque(maxlen=30)` with datetime objects
- **Markov Chains**: 
  - Standard: word → [next_words] (max 50 per word)
  - Context: tuple(4 words) → [next_words]
  - Japanese: separate chain for hiragana/katakana/kanji
- **SQLite LTM**: Full conversation history with emotion vectors, energy, resonance
- **PyTorch State**: Neural model weights + optimizer state + agent genomes

#### 2. **Learning Pipeline**
```python
collect_words() 
    → Grammar Correction (soft, preserves slang)
    → Word Extraction & Cleaning
    → Dynamic Stop-Word Filtering (significance < 0.03)
    → Semantic Classification (emotion/action/object/social/other)
    → Priority Weighting (emotion_boost × resonance_boost × rarity_boost)
    → MultiLangLearner.learn_word() [async, cached translations]
    → Update Markov Chains (standard + context + japanese)
    → Calculate Resonance (Neural Network: 12 features → 512 hidden → 1 output)
    → Train Model (mini-batch, prioritized sampling by message value)
    → Update Agent Rewards (user_interaction, emotion_sync, resonance_match)
    → Save to LTM (batched SQLite + atomic .pt)
```

#### 3. **Agent Evolution**
```
Generation N:
[Agent1(E=80, emoji=✨), Agent2(E=50, emoji=🐾), Agent3(E=-10, emoji=💥)]
                    ↓
         Selection (E >= 80 reproduces)
                    ↓
         Crossover (blend jp_ratio, meme_affinity)
                    ↓
         Mutation (18% rate: new emoji, random jp_ratio)
                    ↓
         Elimination (E <= -20 removed)
                    ↓
Generation N+1:
[Agent1, Agent2, Agent4(mutant, emoji=🌊), Agent5(mutant, emoji=🎲)]
```

**Agent Genome Structure:**
```python
class AgentGenome:
    jp_ratio: float       # 0.05-0.35 (percentage of Japanese words)
    style_emoji: str      # "sparkles", "paw prints", "collision", etc.
    meme_affinity: float  # 0.7-1.3 (weight for meme selection)
```

#### 4. **Resonance Calculation (Neural)**
```python
# Input Features (12-dimensional):
features = [
    lang_sync,          # 1.0 if user lang matches dominant lang
    emotion_sync,       # 1.0 if user emotion matches dominant emotion
    semantic_sync,      # overlap(user_words, top_words) / 3.0
    joy,                # emotion vector components
    tension,
    flow,
    surprise,
    energy,             # sum of word weights
    word_count,         # number of unique words
    time_of_day,        # hour / 24.0
    dummy_1, dummy_2    # padding for embedding dimension
]

# Neural Architecture:
Input(12) → Linear(256) → ReLU → Linear(512) 
         → TransformerMemoryLayer(d_model=512, nhead=8)
         → MultiHeadAttention(512, 8 heads)
         → ResidualBlocks(512 → 512) × 2
         → Linear(512 → 1) → Sigmoid
         → Resonance Score [0..1]
```

#### 5. **Voice Synthesis Pipeline**
```python
Input Text → Language Detection (context buffer, last 10 messages)
           → gTTS Generation (full text, one segment)
           → Pitch Shift (+3 to +6 semitones for female voice)
           → Speed Modulation (1.05-1.15x)
           → Volume Adjustment (-1.5 to +1.5 dB)
           → Low/High-Pass Filters (180Hz - 7kHz)
           → Anime Sighs (30% chance: ふぅ, にゃん, えへ)
           → Master FX Chain:
               • Normalization
               • Dynamic Range Compression (threshold=-20dB, ratio=3.5)
               • Echo/Reverb (25% mix)
               • Grain Synthesis (15% density)
               • Fade In/Out (40ms)
           → Export to OGG (Opus codec, 48kbps)
           → Store in voice_memory dict
           → Send to Telegram
```

---

## 📊 Data Storage

### File Structure
```
YunaNami/
├── yuna.py                  # Main bot code (4500+ lines)
├── yuna_micro.pt            # PyTorch model & memory (5-50MB)
├── yuna_ltm.sqlite          # SQLite conversation DB (grows over time)
├── yuna_data.json           # JSON backup (recent_messages, markov_chain)
├── translation_cache.json   # LRU cache for translations (10k entries)
├── photo_cache/             # User photo storage
│   └── {timestamp}_{id}.jpg
├── reddit_cache/            # Reddit meme cache
├── voice_memory/            # (in-memory dict, not persisted)
├── yuna.log                 # Application logs (INFO level)
└── .env                     # ⚠️ SENSITIVE: Bot token (add to .gitignore!)
```

### Database Schema (SQLite)
```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,                  -- cleaned words
    clean_words TEXT,           -- same as text (legacy)
    user TEXT,                  -- username or first_name
    timestamp REAL,             -- Unix timestamp
    emotion_vector TEXT,        -- JSON: {joy: X, tension: Y, ...}
    energy REAL,                -- sum of word weights
    resonance REAL,             -- neural resonance score [0..1]
    markov_chain TEXT,          -- JSON serialized markov_chain state
    context_chain TEXT,         -- JSON serialized context_chain state
    language TEXT               -- detected language (langdetect)
);

CREATE INDEX idx_timestamp ON messages(timestamp);
CREATE INDEX idx_language ON messages(language);
CREATE INDEX idx_resonance ON messages(resonance);
```

---

## 🎛️ Configuration

### Key Parameters

| Parameter | Default | Description | Tuning Advice |
|-----------|---------|-------------|---------------|
| `MAX_RECENT` | 30 | Recent message buffer size | Increase for longer context |
| `RESO_THRESHOLD` | 20 | Energy trigger for response | Lower = more frequent responses |
| `MAX_AGENTS` | 5 | Maximum agent population | Higher = more diversity, slower evolution |
| `MIN_AGENTS` | 2 | Minimum agent population | Must be >= 1 |
| `CONTEXT_SIZE` | 4 | N-gram context window | Increase for better coherence |
| `RESONANCE_THRESHOLD` | 0.42 | Neural activation threshold | Lower = more sensitive to input |
| `SAVE_INTERVAL` | 30s | Auto-save frequency (JSON) | Balance between safety and I/O |
| `AUTOSAVE_INTERVAL` | 60s | Auto-save frequency (.pt) | PyTorch checkpoints |
| `MAX_MARKOV_PER_WORD` | 50 | Max transitions per word | Limits memory growth |
| `MAX_WORD_ENERGY` | 50 | Energy cap per word | Prevents runaway values |
| `DYNAMIC_STOP_THRESHOLD` | 0.03 | Word significance cutoff | Lower = fewer stop words filtered |
| `MEME_CLEANUP_INTERVAL` | 6h | Cleanup old memes/words | Adjust for memory constraints |

### Environment Variables
```bash
export TELEGRAM_BOT_TOKEN="your_token_here"
export YUMA_NODE_ID="node-001"  # For multi-node resonance sync
export LOG_LEVEL="INFO"          # DEBUG, INFO, WARNING, ERROR
```

### Advanced Configuration (In-Code)
```python
# Multi-Agent Engine
MAE = MultiAgentEngine()
MAE.epsilon = 0.15          # Exploration rate (Q-learning)
MAE.gamma = 0.85            # Discount factor (future rewards)
MAE.alpha = 0.33            # Learning rate (Q-table updates)

# Neural Resonance Model
advanced_resonance_system = AdvancedResonanceSystem(
    input_dim=12,           # Feature vector size
    memory_size=1000,       # Transformer memory capacity
    emo_dim=4,              # Emotion vector size
    hidden_dim=512,         # Hidden layer dimension
    num_heads=4,            # Multi-head attention heads
    attn_dropout=0.15       # Attention dropout rate
)

# Replay Buffer
replay_buffer = ReplayBuffer(maxlen=256)  # Experience replay capacity

# Reddit Fetching
REDDIT_SUBS = [
    'memes', 'dankmemes', 'wholesomememes', 'historymemes', 'Animemes',
    'me_irl', 'surrealmemes', 'ProgrammerHumor', 'japanesememes', 
    'anime_irl', 'memesRU', 'pikabu'  # Add/remove subreddits
]
```

---

## 🔒 Security

### Critical Security Practices

#### 1. **Never Commit Tokens**
```bash
# Add to .gitignore
.env
yuma_micro.pt
yuma_ltm.sqlite
yuma_data.json
translation_cache.json
photo_cache/
reddit_cache/
yuma.log
*.pyc
__pycache__/
```

#### 2. **Use Environment Variables**
```python
import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not set in .env file")
```

#### 3. **Input Validation**
```python
# Already implemented in collect_words():
clean_w = re.sub(r'[^\w]', '', w.lower())
if clean_w and len(clean_w) <= 30:  # Limit word length
    # Process word...
```

#### 4. **Rate Limiting**
- Reddit API: 6 concurrent requests max (semaphore)
- Translation: LRU cache (10k entries) to avoid API abuse
- Web Search: User-Agent header, 15s timeout

#### 5. **Safe Deserialization**
```python
# PyTorch safe globals for custom classes
torch.serialization.add_safe_globals({
    'AgentRandomFlow': AgentRandomFlow,
    'AgentRelevantMeme': AgentRelevantMeme
})
```

#### 6. **Graceful Error Handling**
```python
async def safe_reply_text(message, text, retries=3, timeout=10):
    for attempt in range(retries):
        try:
            await asyncio.wait_for(message.reply_text(text), timeout=timeout)
            return True
        except (asyncio.TimeoutError, telegram.error.TimedOut):
            await asyncio.sleep(2)
    return False
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. **Bot Not Responding**
```bash
# Check logs
tail -f yuma.log

# Verify token
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('TELEGRAM_BOT_TOKEN'))"

# Test network
curl https://api.telegram.org/bot<YOUR_TOKEN>/getMe
```

#### 2. **Memory Errors**
```python
# Reduce buffer sizes in yuma.py
MAX_RECENT = 20              # Default: 30
MAX_MARKOV_PER_WORD = 30     # Default: 50
replay_buffer = ReplayBuffer(maxlen=128)  # Default: 256
```

#### 3. **Slow Performance**
```bash
# Enable DEBUG logging to identify bottlenecks
export LOG_LEVEL=DEBUG

# Profile with cProfile
python -m cProfile -o yuma.prof yuma.py

# Analyze
python -m pstats yuma.prof
> sort cumtime
> stats 20
```

#### 4. **Voice Synthesis Errors**
```bash
# Check FFmpeg installation
ffmpeg -version

# Test gTTS
python -c "from gtts import gTTS; tts = gTTS('test', lang='ja'); tts.save('test.mp3')"

# Check pydub
python -c "from pydub import AudioSegment; print('OK')"
```

#### 5. **Database Locked**
```sql
-- SQLite timeout issues
-- In yuma.py, increase timeout:
conn = sqlite3.connect(LTM_DB_FILE, timeout=30.0)
```

---

## 🧪 Testing

### Unit Tests (TODO)
```bash
# Create tests/ directory structure
tests/
├── test_markov.py
├── test_agents.py
├── test_resonance.py
├── test_voice.py
└── test_memory.py

# Run tests
pytest tests/ -v --cov=yuma
```

### Integration Tests
```python
# tests/test_integration.py
import asyncio
import pytest
from yuma import collect_words, MAE, advanced_resonance_system

@pytest.mark.asyncio
async def test_full_pipeline():
    # Mock Telegram update
    class MockMessage:
        text = "Hello, how are you?"
        photo = None
    
    class MockUpdate:
        message = MockMessage()
        effective_user = type('obj', (object,), {'username': 'test_user'})
    
    # Run collection
    await collect_words(MockUpdate(), None)
    
    # Verify learning
    assert len(markov_chain) > 0
    assert MAE.last_action is not None
```

### Manual Testing Checklist
- [ ] `/start` command shows welcome message
- [ ] Text message triggers learning (check `yuma.log`)
- [ ] Voice message transcription works
- [ ] Photo upload adds to cache
- [ ] `/status` shows correct stats
- [ ] `/evolution` displays agent population
- [ ] Energy threshold triggers `/troll`
- [ ] Reddit fetch adds memes
- [ ] Web search integrates results
- [ ] `.pt` file saves on shutdown
- [ ] SQLite database grows over time

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Development Setup
```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/YunaNami.git
cd YunaNami

# Create feature branch
git checkout -b feature/amazing-feature

# Install dev dependencies
pip install pytest black flake8 mypy

# Run linters
black yuna.py --check
flake8 yuna.py --max-line-length=120
mypy yuna.py --ignore-missing-imports
```

### Contribution Areas

#### High Priority
- 🐛 **Bug fixes**: Race conditions, memory leaks, error handling
- 🔒 **Security**: Input sanitization, API key management, SQL injection prevention
- 🧪 **Testing**: Unit tests for core functions, integration tests, CI/CD pipeline
- 📚 **Documentation**: Docstrings, inline comments, tutorial notebooks

#### Medium Priority
- 🌐 **Language Support**: Additional languages (Spanish, German, Chinese)
- 🎨 **Meme Algorithms**: Template-based generation, GANs, diffusion models
- 🧠 **Neural Architectures**: Transformers, reinforcement learning, meta-learning
- 🔧 **Optimization**: Batching, caching, async improvements, GPU acceleration

#### Experimental
- 🌍 **Multi-Node Resonance**: Production-ready WebSocket server
- 🎙️ **Voice Cloning**: Custom TTS models (Tacotron, FastSpeech)
- 🖼️ **Multimodal**: CLIP integration for image understanding
- 🔗 **Blockchain**: IPFS storage, smart contract persistence

### Code Style

#### PEP 8 Compliance
```python
# Good
async def calculate_resonance_score(user_msg: dict) -> float:
    """
    Calculate resonance score for a user message.
    
    Args:
        user_msg: Dictionary with 'text', 'emotion_vector', 'energy'
    
    Returns:
        Float in range [0.0, 1.0]
    """
    # Implementation...

# Bad
async def calc_res(msg):
    # No docstring, unclear naming
    pass
```

#### Type Hints
```python
from typing import List, Dict, Optional, Tuple

def markov_generate(
    chain: Dict[str, List[str]], 
    start: Optional[str] = None, 
    length: int = 8
) -> List[str]:
    # Implementation...
```

#### Function Length
```python
# Prefer functions under 50 lines
# Extract complex logic into helpers

# Good
async def collect_words(update, context, text=None):
    text = await preprocess_text(update, text)
    words = extract_words(text)
    await update_chains(words)
    await train_model(words)

# Bad - 200+ lines in one function
```

### Pull Request Process

1. **Create Issue First**: Describe bug/feature
2. **Fork & Branch**: `feature/your-feature-name`
3. **Code**: Follow style guide, add tests
4. **Commit**: Use conventional commits (`feat:`, `fix:`, `docs:`)
5. **Test**: Run `pytest`, `black`, `flake8`
6. **PR**: Fill out template, link issue
7. **Review**: Address feedback, iterate
8. **Merge**: Squash commits, update CHANGELOG

---

## 📝 License

This project is licensed under the **MIT License** - see below for details.

```
MIT License

Copyright (c) 2024 0penAGI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## ⚠️ Disclaimer

**This is an experimental research project.** The bot:

- ⚠️ **May generate unpredictable content**: Learns from user input without content filters
- 🔓 **Has no built-in moderation**: Review conversations regularly
- 🌐 **Uses external APIs**: Reddit, RSS, translation services (rate limits apply)
- 💻 **Requires computational resources**: Neural model training (GPU recommended)
- 🧪 **Is not production-ready**: Missing authentication, monitoring, load balancing
- 📊 **Stores conversation data**: SQLite database grows indefinitely
- 🔊 **Generates audio files**: May consume disk space

**Use responsibly and at your own risk.**

### Recommended Safety Measures
1. Run in controlled environment (test group)
2. Monitor logs for inappropriate content
3. Set up backup/restore procedures
4. Implement rate limiting per user
5. Add content filters if deploying publicly
6. Review privacy implications (GDPR, etc.)

---

## 🙏 Acknowledgments

- **python-telegram-bot**: Excellent async Telegram API wrapper ([docs](https://docs.python-telegram-bot.org/))
- **PyTorch**: Deep learning framework ([pytorch.org](https://pytorch.org/))
- **gTTS**: Text-to-speech synthesis ([github](https://github.com/pndurette/gTTS))
- **OpenAI Whisper**: Speech recognition ([github](https://github.com/openai/whisper))
- **Reddit API / PRAW**: Meme content source ([asyncpraw.readthedocs.io](https://asyncpraw.readthedocs.io/))
- **BeautifulSoup4**: HTML parsing for web scraping
- **scikit-learn**: Cosine similarity for meme ranking
- **feedparser**: RSS feed parsing
- **nest_asyncio**: Event loop patching for Jupyter compatibility
- **All contributors**: Thanks to everyone who has contributed code, bug reports, and ideas!

### Special Thanks
- **Community testers**: For finding edge cases and providing feedback
- **Open-source maintainers**: For creating the libraries this project depends on
- **Anime voice synthesis community**: For inspiration on audio processing techniques

---

## 📧 Contact & Support

### GitHub
- **Issues**: [Report bugs or request features](https://github.com/0penAGI/YunaNami/issues)
- **Discussions**: [Join the conversation](https://github.com/0penAGI/YunaNami/discussions)
- **Pull Requests**: [Contribute code](https://github.com/0penAGI/YunaNami/pulls)

### Social
- **Twitter**: [@0penAGI](https://twitter.com/0penAGI)
- **Discord**: [Community Server](https://discord.gg/yunanami) *(coming soon)*
- **Email**: yunanami@0penagi.org

### Getting Help

**Before opening an issue:**
1. Check existing issues for duplicates
2. Review [Troubleshooting](#-troubleshooting) section
3. Include full error traceback
4. Specify Python version, OS, and dependency versions
5. Provide minimal reproducible example

**For general questions:**
- Use [GitHub Discussions](https://github.com/0penAGI/YunaNami/discussions)
- Tag with appropriate label (`question`, `help-wanted`)

---

## 📈 Roadmap

### v3.3 (Q2 2024)
- [ ] **Stable Diffusion Integration**: Full AI meme generation pipeline
  - Text-to-image with SDXL
  - ControlNet for composition
  - LoRA fine-tuning on meme dataset
- [ ] **Advanced Grammar Correction**: Context-aware error fixing
  - Language-specific rules (Russian, English)
  - Preserve slang and colloquialisms
  - Typo detection with edit distance
- [ ] **Voice Cloning**: Personalized synthesis per user
  - YourTTS integration
  - 10-second voice samples
  - Speaker embedding cache
- [ ] **Multimodal Training**: Joint text+image+audio learning
  - CLIP embeddings for images
  - Audio feature extraction (MFCCs)
  - Cross-modal attention
- [ ] **Performance Optimizations**: Production-ready improvements
  - Redis caching for translations
  - PostgreSQL migration from SQLite
  - Horizontal scaling with load balancer
  - Prometheus metrics export

### v4.0 (Q4 2024)
- [ ] **LLM Integration**: GPT-4 / Claude API for natural understanding
  - Hybrid approach: local models + API calls
  - Cost optimization with caching
  - Fallback to Markov chains on API failure
- [ ] **Real-Time Collaboration**: Multi-user features
  - Shared memory across chat groups
  - Collaborative meme creation
  - User reputation system
- [ ] **Mobile App**: iOS/Android companion
  - React Native frontend
  - Push notifications
  - Offline mode with local SQLite
- [ ] **Custom Agent Designer**: Visual UI for creating agents
  - Drag-and-drop genome editor
  - Preview evolution simulation
  - Export/import agent configs
- [ ] **Blockchain Persistence**: Decentralized memory
  - IPFS for large files (images, audio)
  - Ethereum smart contracts for metadata
  - NFT memes (optional, user-controlled)

### v5.0 (Future Vision)
- [ ] **AGI Research**: Advanced cognitive architectures
  - Meta-learning for rapid adaptation
  - Causal reasoning module
  - Episodic memory with attention
- [ ] **Swarm Intelligence**: Multi-bot coordination
  - Distributed resonance network
  - Emergent behavior experiments
  - Inter-bot communication protocol
- [ ] **Quantum Computing**: Hybrid classical-quantum models
  - Quantum annealing for agent selection
  - Variational quantum circuits
  - Proof-of-concept on IBM Quantum

---

## 📊 Performance Benchmarks

### Hardware Recommendations

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| CPU | 2 cores | 4 cores | 8+ cores |
| RAM | 2GB | 4GB | 8GB+ |
| Storage | 5GB | 20GB | 50GB+ SSD |
| GPU | None | GTX 1660 | RTX 3090 |

### Benchmarks (Tested on M1 MacBook Pro, 16GB RAM)

| Operation | Time | Notes |
|-----------|------|-------|
| `collect_words()` (10 words) | 50ms | Without neural training |
| `collect_words()` (10 words, training) | 250ms | With replay buffer updates |
| `troll_text()` (text only) | 100ms | Markov generation |
| `troll_text()` (voice) | 2.5s | gTTS + audio processing |
| `generate_meme()` | 800ms | PIL image manipulation |
| `save_ltm_pt()` | 1.2s | 10MB checkpoint file |
| Reddit fetch (20 memes) | 5s | Async, 6 concurrent requests |
| Web search (5 results) | 3s | DuckDuckGo scraping |
| SQLite batch insert (50 msgs) | 150ms | Optimized bulk insert |

**Bottlenecks:**
- gTTS API calls (network latency: 1-2s)
- PyTorch model training (GPU recommended)
- SQLite writes (use batching)

---

## 🔧 Advanced Configuration

### Custom Agent Classes

Create your own agents by subclassing `AgentInterface`:

```python
class AgentPhilosopher(AgentInterface):
    def __init__(self, name: str, genome: Optional[AgentGenome] = None):
        super().__init__(name, genome)
        self.questions = [
            "でも、本当にそうですか？",
            "Что есть истина?",
            "Why do we exist?"
        ]
    
    async def generate(self, phrase: str) -> str:
        # Add philosophical question
        question = random.choice(self.questions)
        return await self.speak(f"{phrase} {question}", "")

# Register in MAE
MAE.agents.append(AgentPhilosopher("Philosopher"))
```

### Custom Resonance Features

Extend the resonance model with additional features:

```python
def calculate_resonance_score(user_msg: dict) -> float:
    # ... existing code ...
    
    # Add custom feature: message length
    msg_length = len(user_msg.get('text', ''))
    length_score = min(1.0, msg_length / 100.0)
    
    # Add custom feature: time since last message
    time_delta = time.time() - user_msg.get('timestamp', 0)
    recency_score = math.exp(-time_delta / 3600.0)  # 1-hour decay
    
    # Update feature vector
    features.extend([length_score, recency_score])
    
    # ... rest of code ...
```

### WebSocket Resonance Server

Set up a multi-node resonance network:

```python
# server.py
import asyncio
import websockets
import json

connected_nodes = set()

async def handler(websocket, path):
    connected_nodes.add(websocket)
    try:
        async for message in websocket:
            # Broadcast to all nodes except sender
            packet = json.loads(message)
            tasks = [
                ws.send(message) 
                for ws in connected_nodes 
                if ws != websocket
            ]
            await asyncio.gather(*tasks)
    finally:
        connected_nodes.remove(websocket)

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        await asyncio.Future()  # run forever

asyncio.run(main())
```

```bash
# Run server
python server.py

# Connect bots
export YUMA_NODE_ID=node-001
python yuma.py
# In yuma.py, uncomment line ~4600:
# asyncio.create_task(resonance_sync_loop("ws://localhost:8765"))
```

### Custom RSS Feeds

Add your own RSS sources:

```python
CHANNELS = {
    "news": [
        "https://example.com/rss",
        "https://your-blog.com/feed.xml"
    ],
    "custom": [
        "https://api.example.com/rss?category=tech"
    ]
}
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY yuma.py .
COPY .env .

CMD ["python", "yuma.py"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  yuna:
    build: .
    env_file: .env
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

```bash
# Run with Docker
docker-compose up -d

# View logs
docker-compose logs -f yuna

# Stop
docker-compose down
```

---

## 🧬 Research & Papers

### Theoretical Foundations

This project explores several research areas:

#### 1. **Neural Resonance Theory**
- **Hypothesis**: AI systems can achieve "resonance" with users through dynamic attention mechanisms
- **Implementation**: Multi-head attention over emotion vectors + context
- **Related Work**: 
  - Vaswani et al. (2017) - "Attention Is All You Need"
  - Devlin et al. (2018) - "BERT: Pre-training of Deep Bidirectional Transformers"

#### 2. **Evolutionary Multi-Agent Systems**
- **Hypothesis**: Genetic algorithms can optimize agent populations for specific interaction styles
- **Implementation**: Crossover, mutation, fitness-based selection
- **Related Work**:
  - Holland (1992) - "Genetic Algorithms"
  - Sutton & Barto (2018) - "Reinforcement Learning: An Introduction"

#### 3. **Multimodal Learning**
- **Hypothesis**: Joint text+image+audio training improves generalization
- **Implementation**: CLIP embeddings, audio MFCCs, cross-modal attention
- **Related Work**:
  - Radford et al. (2021) - "Learning Transferable Visual Models From Natural Language Supervision"

#### 4. **Self-Organizing Memory**
- **Hypothesis**: Dynamic word significance + context chains enable emergent language understanding
- **Implementation**: Entropy-based significance, N-gram Markov chains
- **Related Work**:
  - Shannon (1948) - "A Mathematical Theory of Communication"
  - Mikolov et al. (2013) - "Efficient Estimation of Word Representations in Vector Space"

### Citations

If you use Yuna Nami in academic research, please cite:

```bibtex
@software{yunanami2024,
  author = {0penAGI},
  title = {Yuna Nami: Neural Chaos AI Chatbot},
  year = {2024},
  url = {https://github.com/0penAGI/YunaNami},
  version = {3.2}
}
```

---

## 🎓 Educational Resources

### Tutorials

#### 1. **Getting Started with Markov Chains**
```python
# Simple Markov text generator
from collections import defaultdict
import random

def build_chain(text, n=2):
    words = text.split()
    chain = defaultdict(list)
    for i in range(len(words) - n):
        key = tuple(words[i:i+n])
        next_word = words[i+n]
        chain[key].append(next_word)
    return chain

def generate(chain, length=10):
    current = random.choice(list(chain.keys()))
    result = list(current)
    for _ in range(length):
        if current not in chain:
            break
        next_word = random.choice(chain[current])
        result.append(next_word)
        current = tuple(result[-len(current):])
    return ' '.join(result)

# Example
text = "the cat sat on the mat the cat ate the rat"
chain = build_chain(text)
print(generate(chain))
```

#### 2. **Building a Simple Q-Learning Agent**
```python
import random

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.Q = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def get_q(self, state, action):
        return self.Q.get((state, action), 0.0)
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = [self.get_q(state, a) for a in self.actions]
        max_q = max(q_values)
        return self.actions[q_values.index(max_q)]
    
    def learn(self, state, action, reward, next_state):
        current_q = self.get_q(state, action)
        max_next_q = max([self.get_q(next_state, a) for a in self.actions])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.Q[(state, action)] = new_q

# Example usage
agent = QLearningAgent(actions=['left', 'right', 'up', 'down'])
state = 'start'
action = agent.choose_action(state)
# ... environment step ...
agent.learn(state, action, reward=1.0, next_state='goal')
```

### Recommended Reading

#### Books
1. **"Speech and Language Processing"** - Jurafsky & Martin
2. **"Deep Learning"** - Goodfellow, Bengio & Courville
3. **"Reinforcement Learning: An Introduction"** - Sutton & Barto
4. **"Genetic Algorithms in Search, Optimization, and Machine Learning"** - Goldberg

#### Papers
1. **Attention Mechanisms**: "Attention Is All You Need" (Vaswani et al., 2017)
2. **Evolutionary Algorithms**: "Genetic Programming" (Koza, 1992)
3. **Multi-Agent Systems**: "Multiagent Systems: A Modern Approach to Distributed AI" (Weiss, 1999)

#### Online Courses
1. **Coursera**: "Natural Language Processing" by DeepLearning.AI
2. **fast.ai**: "Practical Deep Learning for Coders"
3. **MIT OpenCourseWare**: "Artificial Intelligence"

---

## 🌍 Community

### Hall of Fame

**Top Contributors** (by commits):
1. 🥇 [@0penAGI](https://github.com/0penAGI) - Creator & Maintainer
2. 🥈 *Your name here!* - Submit a PR
3. 🥉 *Your name here!* - Submit a PR

**Special Recognition**:
- 🎨 **Best Meme Algorithm**: *Open*
- 🧠 **Best Neural Architecture**: *Open*
- 🐛 **Most Bugs Found**: *Open*
- 📚 **Best Documentation**: *Open*

### Showcase

**Projects Built with Yuna Nami**:
- *Submit yours via PR!*

**Research Papers Citing Yuna Nami**:
- *Submit yours via PR!*

### Events

- 📅 **Monthly Community Call**: First Saturday of each month (Discord)
- 🏆 **Annual Hackathon**: December (details TBA)
- 🎓 **Workshops**: Quarterly tutorials on advanced features

---

## 🔗 Related Projects

### Similar Bots
- [**GPT-Telegram-Bot**](https://github.com/karfly/chatgpt_telegram_bot): OpenAI GPT integration
- [**ChatterBot**](https://github.com/gunthercox/ChatterBot): Rule-based chatbot
- [**Rasa**](https://github.com/RasaHQ/rasa): Production-grade NLU framework

### Inspiration
- [**EleutherAI/gpt-neo**](https://github.com/EleutherAI/gpt-neo): Open-source language models
- [**Stable Diffusion**](https://github.com/CompVis/stable-diffusion): Text-to-image generation
- [**Whisper**](https://github.com/openai/whisper): Speech recognition

### Tools & Libraries
- [**Hugging Face Transformers**](https://github.com/huggingface/transformers): State-of-the-art NLP
- [**LangChain**](https://github.com/hwchase17/langchain): LLM application framework
- [**Weights & Biases**](https://wandb.ai/): Experiment tracking

---

## ❓ FAQ

### General

**Q: Is this a serious AI project or just for fun?**  
A: Both! It's experimental research with playful execution. The neural resonance and evolutionary systems are genuine AI techniques, but the anime voice and chaotic memes are deliberately whimsical.

**Q: Can I use this for commercial projects?**  
A: Yes, under MIT License. Attribution appreciated but not required. Add your own content moderation and safety measures.

**Q: Why "Yuna Nami"?**  
A: "Yuna" (ユナ) + "Nami" (波 = wave). Represents the flow of information and resonance between AI and users.

### Technical

**Q: How does the bot learn?**  
A: Three layers:
1. **Markov chains**: Statistical word patterns
2. **Neural resonance**: Emotion/context matching
3. **Multi-agent evolution**: Behavioral adaptation

**Q: Does it really understand language?**  
A: No, it's statistical pattern matching + attention mechanisms. Not true language understanding (like GPT), but emergent coherence from data.

**Q: Can it remember conversations forever?**  
A: SQLite LTM stores indefinitely, but active memory (recent_messages) is limited to 30 messages. Implement retention policies if privacy matters.

**Q: Why PyTorch instead of TensorFlow?**  
A: Personal preference. PyTorch is more Pythonic and easier for research. TensorFlow version is feasible (contributions welcome!).

### Usage

**Q: Can I run multiple bots with shared memory?**  
A: Yes! Use the resonance sync protocol (WebSocket server). Each bot broadcasts its state, and all bots converge on shared resonance field.

**Q: How much does it cost to run?**  
A: **Free** if self-hosted. Costs:
- Server: $5-20/month (DigitalOcean, AWS)
- APIs: gTTS (free), Reddit (free), translations (free via Google Translate)
- Optional: OpenAI API ($0.002/1k tokens)

**Q: Can I deploy on Heroku/Replit/etc.?**  
A: Yes, but:
- Heroku: Need worker dyno (not web)
- Replit: Works but may hit memory limits
- Railway: Good choice (persistent storage)
- Fly.io: Excellent (free tier)

---

## 📜 Changelog

### v3.2 (Current - 2024-01-15)
- ✨ Advanced Resonance System with Multi-Head Attention
- 🧬 Genetic Algorithm for Multi-Agent Evolution
- 🗣️ Improved anime voice synthesis (female pitch, sighs)
- 🌐 Web search integration (DuckDuckGo)
- 📊 SQLite LTM with batched writes
- 🔒 Security improvements (token management, input validation)
- 🐛 Bug fixes: race conditions, memory leaks, error handling

### v3.1 (2023-12-01)
- 🧠 Neural resonance model (PyTorch)
- 🎙️ Whisper transcription
- 📰 RSS feed aggregation
- 🎨 Multi-language meme generation

### v3.0 (2023-10-15)
- 🔄 Full async/await refactor
- 🤖 Multi-agent Q-learning system
- 🗃️ SQLite long-term memory
- 🎭 Agent evolution with genomes

### v2.0 (2023-08-01)
- 🗣️ Voice synthesis (gTTS)
- 📸 Meme generation
- 🌐 Reddit integration
- 📚 Markov chains

### v1.0 (2023-06-01)
- 🚀 Initial release
- 💬 Basic Telegram bot
- 📝 Text learning

---

<div align="center">

## 🎉 Thank You!

**Yuna Nami wouldn't exist without the open-source community.**

If you've made it this far, you're awesome! 🌟

### Support the Project

- ⭐ **Star** this repo
- 🐛 **Report** bugs
- 💡 **Share** ideas
- 🔀 **Contribute** code
- 📣 **Spread** the word

---

**Made with ❤️ and ☕ by [0penAGI](https://github.com/0penAGI)**

*"In chaos, we find resonance. In resonance, we find truth." - Yuna Nami*

</div>
