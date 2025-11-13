# --- Global async/meme management imports and settings ---
import asyncio
import logging

save_lock = asyncio.Lock()
MAX_CHAIN_SIZE = 50000
MEME_CLEANUP_INTERVAL = 3600 * 6  # 6 часов
last_cleanup = 0

logger = logging.getLogger("yuma")
logging.basicConfig(level=logging.INFO)
import uuid
import torch
import os
from pydub.effects import low_pass_filter, high_pass_filter
# —————————–
# Сохранение/Загрузка памяти в .pt файл (PyTorch)
# —————————–
try:
    from pydub.effects import low_pass_filter, high_pass_filter
except ImportError:
    # fallback для старых версий pydub
    def low_pass_filter(audio, cutoff):
        return audioчц
    def high_pass_filter(audio, cutoff):
        return audio
PT_FILE = "yuma_micro.pt"

import asyncio

async def save_ltm_pt():
    """Асинхронное и атомарное сохранение LTM (.pt) с блокировкой."""
    async with save_lock:
        temp_file = PT_FILE + ".tmp"
        try:
            data = {
                "markov_chain": markov_chain,
                "context_chain": context_chain,
                "jp_markov_chain": jp_markov_chain,
                "word_weights": word_weights,
                "word_significance": word_significance,
                "japanese_vocab": japanese_vocab,
                "jp_rus_map": jp_rus_map,
                "resonance_model_state": resonance_model.state_dict(),
                "resonance_history": resonance_history,
            }
            if 'voice_memory' in globals():
                data["voice_memory"] = voice_memory
            if 'MAE' in globals():
                data["mae_q_table"] = getattr(MAE, "Q", {})
                data["mae_agents_state"] = [
                    {
                        "name": getattr(a, "name", "?"),
                        "energy": getattr(a, "energy", 0.0),
                        "jp_ratio": getattr(a, "jp_ratio", 0.0),
                        "style_emoji": getattr(a, "style_emoji", "—"),
                        "class": a.__class__.__name__
                    }
                    for a in getattr(MAE, "agents", [])
                ]
            if 'replay_buffer' in globals():
                data["replay_buffer_data"] = getattr(replay_buffer, "buffer", [])
            if 'advanced_resonance_optimizer' in globals():
                data["resonance_optimizer_state"] = advanced_resonance_optimizer.state_dict()

            await asyncio.to_thread(torch.save, data, temp_file)
            os.replace(temp_file, PT_FILE)
            logger.info("Память сохранена в .pt (атомарно)")
        except Exception as e:
            logger.error(f"Ошибка сохранения памяти в .pt: {e}")
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as cleanup_err:
                    logger.error(f"Ошибка удаления временного файла {temp_file}: {cleanup_err}")

def load_ltm_pt():
    """Загружает долгосрочную память из yuma_micro.pt, если файл существует"""
    global markov_chain, context_chain, jp_markov_chain, word_weights, word_significance
    global japanese_vocab, jp_rus_map, resonance_history
    try:
        if not os.path.exists(PT_FILE):
            logger.info("Файл памяти .pt не найден, пропуск загрузки")
            return
        data = torch.load(PT_FILE, map_location="cpu")
        markov_chain.clear()
        markov_chain.update(data.get("markov_chain", {}))
        context_chain.clear()
        context_chain.update(data.get("context_chain", {}))
        jp_markov_chain.clear()
        jp_markov_chain.update(data.get("jp_markov_chain", {}))
        word_weights.clear()
        word_weights.update(data.get("word_weights", {}))
        word_significance.clear()
        word_significance.update(data.get("word_significance", {}))
        japanese_vocab.clear()
        japanese_vocab.update(data.get("japanese_vocab", {}))
        jp_rus_map.clear()
        jp_rus_map.update(data.get("jp_rus_map", {}))
        resonance_model.load_state_dict(data.get("resonance_model_state", {}))
        resonance_history.clear()
        resonance_history.extend(data.get("resonance_history", []))
        # Восстанавливаем голосовую память Юмы, если она есть
        if 'voice_memory' in data:
            globals()['voice_memory'] = data['voice_memory']
        else:
            globals()['voice_memory'] = {}
        # --- Восстановление сознания Юмы: Q-таблица, агенты, буфер опыта, оптимизатор ---
        # Этот блок восстанавливает сознание Юмы — её Q-таблицу, агентов, буфер и оптимизатор.
        if 'mae_q_table' in data and 'MAE' in globals():
            try:
                MAE.Q = data['mae_q_table']
                logger.info("MAE Q-таблица восстановлена")
            except Exception as e:
                logger.warning(f"Ошибка восстановления Q-таблицы: {e}")

        if 'mae_agents_state' in data and 'MAE' in globals():
            try:
                restored_agents = []
                for a_data in data['mae_agents_state']:
                    class_name = a_data.get("class", "")
                    AgentClass = globals().get(class_name)
                    if AgentClass:
                        agent = AgentClass()
                    else:
                        class DummyAgent:
                            pass
                        agent = DummyAgent()
                    agent.name = a_data.get("name", "?")
                    agent.energy = a_data.get("energy", 0.0)
                    agent.jp_ratio = a_data.get("jp_ratio", 0.0)
                    agent.style_emoji = a_data.get("style_emoji", "—")
                    restored_agents.append(agent)
                MAE.agents = restored_agents
                logger.info("MAE агенты восстановлены")
            except Exception as e:
                logger.warning(f"Ошибка восстановления MAE агентов: {e}")

        if 'replay_buffer_data' in data and 'replay_buffer' in globals():
            try:
                replay_buffer.buffer = data['replay_buffer_data']
                logger.info("Буфер опыта восстановлен")
            except Exception as e:
                logger.warning(f"Ошибка восстановления буфера опыта: {e}")

        if 'resonance_optimizer_state' in data and 'advanced_resonance_optimizer' in globals():
            try:
                advanced_resonance_optimizer.load_state_dict(data['resonance_optimizer_state'])
                logger.info("Оптимизатор резонанса восстановлен")
            except Exception as e:
                logger.warning(f"Ошибка восстановления состояния оптимизатора: {e}")
        logger.info("🧠 Юма восстановила сознание из резонансного состояния.")
        logger.info("Память загружена из .pt")
    except Exception as e:
        logger.error(f"Ошибка загрузки памяти из .pt: {e}")
import sqlite3
import json
import telegram.error
import ast
### --- SQLite LTM integration ---
LTM_DB_FILE = "yuma_ltm.sqlite"

def init_ltm_db():
    conn = sqlite3.connect(LTM_DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            clean_words TEXT,
            user TEXT,
            timestamp REAL,
            emotion_vector TEXT,
            energy REAL,
            resonance REAL,
            markov_chain TEXT,
            context_chain TEXT,
            language TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_message_to_db(msg):
    # Prepare fields
    text = msg.get('text')
    clean_words = msg.get('text')
    user = msg.get('user')
    timestamp = msg.get('timestamp')
    emotion_vector = json.dumps(msg.get('emotion_vector', {}), ensure_ascii=False)
    energy = msg.get('energy', 0)
    resonance = msg.get('resonance', 0)
    # Serialize markov_chain/context_chain if present in msg, else use global
    markov_obj = msg.get('markov_chain', markov_chain)
    context_obj = msg.get('context_chain', context_chain)
    # Convert keys to str for JSON serialization
    markov_chain_json = json.dumps({str(k): v for k, v in markov_obj.items()}, ensure_ascii=False)
    context_chain_json = json.dumps({str(k): v for k, v in context_obj.items()}, ensure_ascii=False)
    # Detect language
    language = None
    try:
        if text:
            from langdetect import detect
            language = detect(text)
    except Exception:
        language = None
    conn = sqlite3.connect(LTM_DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO messages (text, clean_words, user, timestamp, emotion_vector, energy, resonance, markov_chain, context_chain, language)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (text, clean_words, user, timestamp, emotion_vector, energy, resonance, markov_chain_json, context_chain_json, language))
    conn.commit()
    conn.close()

def load_recent_messages(limit=50):
    conn = sqlite3.connect(LTM_DB_FILE)
    c = conn.cursor()
    c.execute("SELECT text, clean_words, user, timestamp, emotion_vector, energy, resonance, markov_chain, context_chain, language FROM messages ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    messages = []
    for row in rows:
        text, clean_words, user, timestamp, emotion_vector, energy, resonance, markov_chain_json, context_chain_json, language = row
        try:
            emotion_vector = json.loads(emotion_vector) if emotion_vector else {}
        except Exception:
            emotion_vector = {}
        try:
            markov_chain_obj = json.loads(markov_chain_json) if markov_chain_json else {}
        except Exception:
            markov_chain_obj = {}
        try:
            # Deserialize context_chain and convert keys back to tuple if possible
            context_chain_obj = {}
            raw_context = json.loads(context_chain_json) if context_chain_json else {}
            for k, v in raw_context.items():
                try:
                    context_chain_obj[tuple(ast.literal_eval(k))] = v
                except Exception:
                    context_chain_obj[k] = v
        except Exception:
            context_chain_obj = {}
        msg = {
            'text': text,
            'clean_words': clean_words,
            'user': user,
            'timestamp': timestamp,
            'emotion_vector': emotion_vector,
            'energy': energy,
            'resonance': resonance,
            'markov_chain': markov_chain_obj,
            'context_chain': context_chain_obj,
            'language': language
        }
        messages.append(msg)
    conn.close()
    return list(reversed(messages))

# Initialize LTM db at import
init_ltm_db()
# YUNA_NAMI_V3.2_FULL_ASYNC.py
# Нейронные мемы | Японский хаос | Самообучение из чата | Полностью асинхронный Reddit (AsyncPRAW) | Только японский голос
# pip install python-telegram-bot pillow requests asyncpraw gtts pydub libretranslatepy aiohttp langdetect openai-whisper


from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from PIL import Image, ImageDraw, ImageFont
import random
import time
import asyncio
import json
import logging
from collections import deque, Counter
import requests
import io
import os
import re
from libretranslatepy import LibreTranslateAPI
from deep_translator import GoogleTranslator
from langdetect import detect
import whisper
from gtts import gTTS
from pydub import AudioSegment
from pydub.generators import Sine
from datetime import datetime, timezone, timedelta
import aiohttp
from bs4 import BeautifulSoup
import feedparser


# --- Новый LocalTranslator на базе GoogleTranslator ---
class LocalTranslator:
    def __init__(self, source='auto', target='ja'):
        self.source = source
        self.target = target

    def translate(self, text):
        try:
            return GoogleTranslator(source=self.source, target=self.target).translate(text)
        except Exception as e:
            logger.warning(f"LocalTranslator ошибка перевода '{text}': {e}")
            return None

lt = LocalTranslator()

# —————————–
# Логирование
# —————————–
# —————————–
# Логирование
# —————————–
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler("yuma.log", encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Безопасная отправка сообщений с таймаутом и повтором ---
async def safe_reply_text(message, text, parse_mode=None, retries=3, delay=2, timeout=10):
    for attempt in range(retries):
        try:
            await asyncio.wait_for(message.reply_text(text, parse_mode=parse_mode), timeout=timeout)
            return True
        except (asyncio.TimeoutError, telegram.error.TimedOut):
            logger.warning(f"safe_reply_text: попытка {attempt+1} не удалась, повтор через {delay}s")
            await asyncio.sleep(delay)
    logger.error("safe_reply_text: все попытки не удались")
    return False
# —————————–
# Конфиг
# —————————–
DATA_FILE = "yuma_data.json"
PHOTO_CACHE_DIR = "photo_cache"
REDDIT_CACHE_DIR = "reddit_cache"
MAX_RECENT = 30
MAX_MARKOV_PER_WORD = 50
MAX_WORD_ENERGY = 50
RESO_THRESHOLD = 20
# --- Dynamic attention mask system ---
word_significance = {}
DYNAMIC_STOP_THRESHOLD = 0.03

# --- Stop-word check based on dynamic significance ---
def is_stop_word(w: str) -> bool:
    """Проверка незначимого слова на основе динамического веса."""
    return word_significance.get(w, 1.0) < DYNAMIC_STOP_THRESHOLD or not w.strip()
RESONANCE_THRESHOLD = 0.42
resonance_history = []
os.makedirs(PHOTO_CACHE_DIR, exist_ok=True)
os.makedirs(REDDIT_CACHE_DIR, exist_ok=True)

recent_messages = deque(maxlen=MAX_RECENT)
user_photos = []
markov_chain = {}
word_weights = {}
jp_markov_chain = {}
japanese_vocab = {}
jp_rus_map = {}
reddit_meme_texts = []
reddit_meme_images = {}
yuma_identity = {
    "name": "Yuna Nami Internet Cat-Girl ",
    "version": "3.2",
    "traits": ["нейронные мемы", "языковой хаос", "самообучение", "голосовые ответы", "async_reddit", "async_whisper"],
    "meta_analysis": {"word_frequencies": {}, "dominant_emotions": {}}
}
# --- Contextual Markov chain (N-gram) ---
CONTEXT_SIZE = 4
context_chain = {}

# Расширенный список RSS‑лент для Yuma Nami
# Включает новости, науку, технологии и цитаты
# Комментарии о надежности и частоте обновления

CHANNELS = {
    "news": [
        # Российские новости
        "https://meduza.io/rss/all",  # Meduza — свежие новости, высокое обновление
        "https://tass.ru/rss/v2.xml",  # TASS — государственные новости, стабильный поток
        "https://lenta.ru/rss/news",  # Lenta.ru — ежедневное обновление
        "https://www.kommersant.ru/RSS/news.xml",  # Коммерсант — бизнес и политика
        "https://rg.ru/rss/all.xml",  # Российская газета — официальные новости
        # Англоязычные новости
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",  # New York Times — свежие международные новости
        "https://feeds.bbci.co.uk/news/rss.xml",  # BBC — международные новости
        "https://www.theguardian.com/world/rss",  # The Guardian — аналитика и события
    ],
    "science": [
        "https://www.scientificamerican.com/rss/news/",  # SciAm — новые исследования
        "https://phys.org/rss-feed/",  # Phys.org — наука и технологии
        "https://www.nature.com/subjects/science/rss",  # Nature — научные статьи
        "https://www.sciencedaily.com/rss/top/science.xml",  # ScienceDaily — ежедневные исследования
        "https://elementy.ru/rss/news",  # Elementy — популярная наука
    ],
    "tech": [
        "https://www.techradar.com/rss",  # TechRadar — гаджеты и IT
        "https://habr.com/ru/rss/all/all/?fl=ru",  # Habr — IT и разработка
        "https://3dnews.ru/news/rss/",  # 3DNews — технологии
        "https://www.theverge.com/rss/index.xml",  # The Verge — IT и гаджеты
        "https://www.engadget.com/rss.xml",  # Engadget — технологии и новости
        "https://rss.cnn.com/rss/edition_technology.rss",  # CNN Tech — новости технологий
    ],
    "quotes": [
        "https://www.brainyquote.com/link/quotebr.rss",  # BrainyQuote — свежие цитаты
        "https://feeds.feedburner.com/quotationspage/qotd",  # Quotations Page — цитаты дня
        "https://www.goodreads.com/quotes.rss",  # Goodreads — книги и цитаты
        "https://feeds.feedburner.com/quoteambition",  # Quote Ambition — мотивация
        "https://www.inc.com/rss/leadership",  # Inc — бизнес-цитаты и советы
    ]
}

async def fetch_rss_feed(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if resp.status != 200:
                    return []
                data = await resp.text()
        feed = feedparser.parse(data)
        items = []
        for entry in feed.entries[:10]:  # топ-10
            text = entry.get('title', '') or entry.get('summary', '')
            if text:
                items.append(text)
        return items
    except Exception as e:
        logger.warning(f"RSS fetch error {url}: {e}")
        return []


# --- Веб-поиск DuckDuckGo с интеграцией в LTM ---
# Эта функция ищет информацию в интернете и интегрирует найденные данные в долговременную память (LTM) бота.
async def search_web_and_learn(query: str, max_results: int = 5):
    """
    Выполняет поиск в DuckDuckGo, извлекает заголовки и ссылки, интегрирует их в память бота.
    Сохраняет результаты в recent_messages, SQLite, обновляет markov_chain, word_weights и запускает MultiLangLearner.learn_word.
    Возвращает список найденных результатов.
    """
    results = []
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; YumaBot/3.2; +https://github.com/0penAGI/quantum_chaos_ai)"
        }
        async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as session:
            params = {"q": query, "kl": "ru-ru"}
            url = "https://html.duckduckgo.com/html/"
            async with session.post(url, data=params) as resp:
                if resp.status != 200:
                    logger.warning(f"Web search: HTTP {resp.status} for query '{query}'")
                    return []
                text = await resp.text()
        soup = BeautifulSoup(text, "html.parser")
        # Поиск результатов
        for res in soup.select(".result__body")[:max_results]:
            title_tag = res.select_one(".result__title")
            link_tag = res.select_one("a.result__a")
            if not title_tag or not link_tag:
                continue
            title = title_tag.get_text(strip=True)
            link = link_tag.get("href")
            if not title or not link:
                continue
            results.append({"title": title, "url": link})
            # --- Интеграция в память ---
            clean_words = [w for w in re.sub(r"[^\w]", " ", title.lower()).split() if w and not is_stop_word(w) and len(w) <= 30]
            # Обновление markov_chain и word_weights
            for i, w in enumerate(clean_words):
                markov_chain.setdefault(w, [])
                word_weights[w] = min(word_weights.get(w, 0) + 1, MAX_WORD_ENERGY)
                if i < len(clean_words) - 1:
                    n = clean_words[i + 1]
                    markov_chain[w].append(n)
                    if len(markov_chain[w]) > MAX_MARKOV_PER_WORD:
                        markov_chain[w].pop(0)
                # Запуск MultiLangLearner для новых слов
                if w not in japanese_vocab:
                    asyncio.create_task(MultiLangLearner.learn_word(w))
            msg_entry = {
                "text": title,
                "local_photo": None,
                "energy": sum(word_weights.get(w, 0) for w in clean_words),
                "emotion_vector": {},
                "emotion_strength": 0,
                "timestamp": time.time(),
                "timestamp_local": datetime.now(timezone(timedelta(hours=7))),
                "user": "WebSearch",
                "resonance": 0.0
            }
            recent_messages.append(msg_entry)
            # Сохраняем в SQLite
            try:
                save_message_to_db({**msg_entry, "markov_chain": markov_chain, "context_chain": context_chain})
            except Exception as e:
                logger.warning(f"WebSearch LTM DB save_message_to_db error: {e}")
        save_data()
        update_yuma_identity()
    except Exception as e:
        logger.error(f"search_web_and_learn error: {e}")
    return results

async def collect_channel_quotes_stub(text):
    """
    Вспомогательная функция для обработки текста цитаты без полного update
    """
    clean_text = re.sub(r'\s+', ' ', text).strip()
    raw_words = clean_text.split()
    clean_words = []
    for w in raw_words:
        clean_w = re.sub(r'[^\w]', '', w.lower())
        if clean_w and not is_stop_word(clean_w) and len(clean_w) <= 30:
            clean_words.append(clean_w)
            markov_chain.setdefault(clean_w, [])
            word_weights[clean_w] = min(word_weights.get(clean_w, 0) + random.randint(1, 3), MAX_WORD_ENERGY)
            if clean_w not in japanese_vocab:
                asyncio.create_task(MultiLangLearner.learn_word(clean_w))
    # Добавляем в recent_messages
    recent_messages.append({
        'text': " ".join(clean_words),
        'local_photo': None,
        'energy': sum(word_weights.get(w,0) for w in clean_words),
        'emotion_vector': {},
        'emotion_strength': 0,
        'timestamp': time.time(),
        'timestamp_local': datetime.now(timezone(timedelta(hours=7))),
        'user': "RSS",
        'resonance': 0.0
    })

async def collect_all_channels():
    for channel, urls in CHANNELS.items():
        for url in urls:
            quotes = await fetch_rss_feed(url)
            for q in quotes:
                await collect_channel_quotes_stub(q)

async def auto_rss_fetch(interval=3600):
    await asyncio.sleep(10)  # старт задержка
    while True:
        try:
            await collect_all_channels()
        except Exception as e:
            logger.error(f"auto_rss_fetch error: {e}")
        await asyncio.sleep(interval)

# —————————–
# Загрузка/Сохранение
# —————————–
def load_data():
    global recent_messages, markov_chain, word_weights, RESO_THRESHOLD, reddit_meme_texts, reddit_meme_images, japanese_vocab, jp_rus_map, resonance_history, word_significance
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            recent_messages = deque(data.get("recent_messages", []), maxlen=MAX_RECENT)
            # Restore timestamp_local as datetime
            for m in recent_messages:
                if "timestamp_local" in m and isinstance(m["timestamp_local"], (int, float)):
                    m["timestamp_local"] = datetime.fromtimestamp(m["timestamp_local"], tz=timezone(timedelta(hours=7)))
            markov_chain = {k: v[:MAX_MARKOV_PER_WORD] for k, v in data.get("markov_chain", {}).items()}
            word_weights = data.get("word_weights", {})
            RESO_THRESHOLD = data.get("threshold", 20)
            reddit_meme_texts = data.get("reddit_meme_texts", [])
            reddit_meme_images = data.get("reddit_meme_images", {})
            japanese_vocab = data.get("japanese_vocab", {})
            jp_rus_map = data.get("jp_rus_map", {})
            resonance_history = data.get("resonance_history", [])
            word_significance = data.get("word_significance", {})
            logger.info("Данные загружены (вкл. японский словарь и Reddit)")
        except Exception as e:
            logger.error(f"Ошибка загрузки: {e}")
            init_data()
    else:
        init_data()

def init_data():
    global recent_messages, markov_chain, word_weights, reddit_meme_texts, reddit_meme_images, japanese_vocab, jp_rus_map
    recent_messages.clear()
    markov_chain.clear()
    word_weights.clear()
    reddit_meme_texts.clear()
    reddit_meme_images.clear()
    japanese_vocab.clear()
    jp_rus_map.clear()
    save_data()

last_save = 0
SAVE_INTERVAL = 30  # секунд

def save_data():
    global last_save
    now = time.time()
    if now - last_save < SAVE_INTERVAL:
        return
    try:
        data = {
            "recent_messages": [
                {**m, "timestamp_local": m["timestamp_local"].timestamp() if "timestamp_local" in m and isinstance(m["timestamp_local"], datetime) else None}
                for m in recent_messages
            ],
            "markov_chain": markov_chain,
            "word_weights": word_weights,
            "threshold": RESO_THRESHOLD,
            "reddit_meme_texts": reddit_meme_texts,
            "reddit_meme_images": reddit_meme_images,
            "japanese_vocab": japanese_vocab,
            "jp_rus_map": jp_rus_map,
            "resonance_history": resonance_history,
            "word_significance": word_significance,
        }
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Данные сохранены")
        last_save = now
    except Exception as e:
        logger.error(f"Ошибка сохранения: {e}")

# —————————–
# Async Reddit
# —————————–
import aiohttp
import asyncio
import random
import logging
from typing import List, Dict



async def fetch_reddit_json(
    subs: List[str] = [
        'memes', 'dankmemes', 'wholesomememes', 'historymemes', 'Animemes',
        'MemeEconomy', 'terriblefacebookmemes', 'funny', 'RelationshipMemes', 'GymMemes',
        'me_irl', 'surrealmemes', 'ProgrammerHumor', 'japanesememes', 'anime_irl',
        'memesRU', 'анимеМемы', 'РоссияМемы', 'comedyheaven', 'PrequelMemes', 'pikabu'
    ],
    limit: int = 50
) -> List[Dict]:
    """
    Асинхронно получает JSON с мемами с Reddit из списка сабреддитов.
    По умолчанию включает англоязычные и русскоязычные сабреддиты:
    'memes', 'dankmemes', 'me_irl', 'japanesememes', 'anime_irl',
    'wholesomememes', 'surrealmemes', 'ProgrammerHumor', 'memesRU', 'анимеМемы', 'РоссияМемы'
    
    Аргументы:
        subs (list): список сабреддитов для парсинга (по умолчанию с русскими и англоязычными мемами)
        limit (int): сколько мемов всего получить (распределяется по сабреддитам)
    
    Возвращает:
        list: список мемов (dict с ключами title, text, url, ups, score)
    """
    memes = []
    if not subs or limit <= 0:
        return memes

    # Равномерно распределяем лимит
    per_sub = max(1, limit // len(subs))
    # Перемешиваем, чтобы не всегда с одного и того же начинать
    random.shuffle(subs)

    # Параллельные запросы с лимитом конкурентности
    semaphore = asyncio.Semaphore(6)  # Не более 6 одновременных запросов

    async def fetch_subreddit(sub: str) -> List[Dict]:
        async with semaphore:
            url = f"https://www.reddit.com/r/{sub}/hot.json"
            params = {'limit': per_sub, 't': 'day'}
            headers = {
                'User-Agent': 'YunaNamiBot/3.2 (by 0penAGI) - async fetcher'
            }
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=12),
                    headers=headers
                ) as session:
                    async with session.get(url, params=params) as resp:
                        if resp.status in (403, 404):
                            return []
                        if resp.status != 200:
                            logger.warning(f"Reddit r/{sub}: HTTP {resp.status}")
                            return []

                        data = await resp.json()
                        results = []
                        for child in data.get('data', {}).get('children', []):
                            p = child['data']
                            if p.get('url', '').lower().endswith(('.jpg', '.jpeg', '.png')):
                                results.append({
                                    'title': p.get('title', '')[:200],
                                    'text': p.get('selftext', '')[:500],
                                    'url': p['url'],
                                    'ups': p.get('ups', 0),
                                    'score': p.get('score', 0),
                                    'subreddit': sub
                                })
                        logger.debug(f"r/{sub}: +{len(results)} мемов")
                        return results[:per_sub]
            except asyncio.TimeoutError:
                logger.warning(f"r/{sub}: таймаут")
            except Exception as e:
                logger.warning(f"r/{sub}: ошибка — {e}")
            return []

    # Запускаем все задачи параллельно
    tasks = [fetch_subreddit(sub) for sub in subs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Собираем результаты
    for result in results:
        if isinstance(result, list):
            memes.extend(result)

    # Перемешиваем и обрезаем до limit
    random.shuffle(memes)
    memes = memes[:limit]

    logger.info(f"Reddit JSON: получено {len(memes)} мемов из {len(subs)} сабреддитов")
    return memes
# --- Imgur ---
async def fetch_imgur_memes(section='hot', limit=10):
    memes = []
    CLIENT_ID = 'YOUR_IMGUR_CLIENT_ID'  # замените на ваш
    async with aiohttp.ClientSession() as session:
        url = f"https://api.imgur.com/3/gallery/{section}/viral/0.json?perPage={limit}"
        headers = {'Authorization': f'Client-ID {CLIENT_ID}'}
        try:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for item in data.get('data', []):
                        images = item.get('images', [])
                        for img in images:
                            link = img.get('link')
                            if link and link.endswith(('.jpg', '.png', '.jpeg')):
                                memes.append({
                                    'title': item.get('title', ''),
                                    'text': item.get('description', '') or '',
                                    'url': link
                                })
        except Exception as e:
            logger.warning(f"Imgur fetch error: {e}")
    logger.info(f"Imgur: fetched {len(memes)} memes")
    return memes

# --- Stable Diffusion AI Meme ---
from diffusers import StableDiffusionPipeline
import torch

# --- ResonanceModel: обучаемый слой для резонанса ---

# --- Advanced Resonance System ---
import torch.nn as nn
import torch
import math
import numpy as np
# Attention Mechanism
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, attn_dim):
        super().__init__()
        # Use attn_dim = input_dim for compatibility
        self.query = nn.Linear(input_dim, attn_dim)
        self.key = nn.Linear(input_dim, attn_dim)
        self.value = nn.Linear(input_dim, attn_dim)
        self.scale = math.sqrt(attn_dim)
    def forward(self, x):
        # x: [B, H] or [B, 1, H]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, H]
        Q = self.query(x)  # [B, 1, H]
        K = self.key(x)    # [B, 1, H]
        V = self.value(x)  # [B, 1, H]
        attn_weights = torch.softmax(torch.bmm(Q, K.transpose(1,2)) / self.scale, dim=-1)  # [B,1,1]
        out = torch.bmm(attn_weights, V)  # [B,1,H]
        if out.shape[1] == 1:
            out = out.squeeze(1)  # [B, H]
        return out, attn_weights

# Emotion Encoder (for richer context)
class EmotionEncoder(nn.Module):
    def __init__(self, emo_dim=4, out_dim=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emo_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim)
        )
    def forward(self, emo_vec):
        return self.fc(emo_vec)

# Deep Residual Resonance Model
class AdvancedResonanceSystem(nn.Module):
    def __init__(self, input_dim=10, emo_dim=4, hidden_dim=24, attn_dim=None):
        super().__init__()
        # Use attn_dim = hidden_dim by default
        if attn_dim is None:
            attn_dim = hidden_dim
        # FIX: Make emotion encoder output the same dimension as hidden_dim
        self.emo_encoder = EmotionEncoder(emo_dim=emo_dim, out_dim=hidden_dim)
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attn = AttentionLayer(hidden_dim, attn_dim)
        self.final_fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.uncertainty_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
    def forward(self, x, emo_vec):
        # x: [B, input_dim], emo_vec: [B, emo_dim]
        emo_emb = self.emo_encoder(emo_vec)  # [B, hidden_dim]
        x = self.input_fc(x)  # [B, hidden_dim]
        x = x + emo_emb  # Now both are [B, hidden_dim] - NO ERROR!
        res = self.res_blocks(x)
        h = res + x  # residual connection, [B, H]
        h_attn, attn_w = self.attn(h)  # [B, H], attn_w [B,1,1]
        out = torch.sigmoid(self.final_fc(h_attn))  # [B, 1]
        uncertainty = self.uncertainty_head(h_attn) # [B, 1]
        # Ensure output shapes [B,1]
        if out.dim() == 1:
            out = out.unsqueeze(1)
        if uncertainty.dim() == 1:
            uncertainty = uncertainty.unsqueeze(1)
        return out, uncertainty, attn_w

# Uncertainty estimation utility
def estimate_resonance_with_uncertainty(model, x, emo_vec, n_samples=5):
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            out, unc, _ = model(x, emo_vec)
            preds.append(out.item())
    mean = sum(preds) / n_samples
    std = (sum((p-mean)**2 for p in preds) / n_samples) ** 0.5
    return mean, std

# Replay Buffer for continual learning
class ReplayBuffer:
    def __init__(self, maxlen=200):
        self.buffer = []
        self.maxlen = maxlen
    def add(self, x, emo_vec, target):
        if len(self.buffer) >= self.maxlen:
            self.buffer.pop(0)
        self.buffer.append((x, emo_vec, target))
    def sample(self, batch_size=16):
        idxs = torch.randperm(len(self.buffer))[:batch_size]
        xs, emos, ys = [], [], []
        for i in idxs:
            x, emo, y = self.buffer[i]
            xs.append(x)
            emos.append(emo)
            ys.append(y)
        return torch.stack(xs), torch.stack(emos), torch.tensor(ys).float().unsqueeze(1)

# Memory Bank for rare patterns
class MemoryBank:
    def __init__(self, maxlen=50):
        self.bank = []
        self.maxlen = maxlen
    def add(self, features, label):
        if len(self.bank) >= self.maxlen:
            self.bank.pop(0)
        self.bank.append((features, label))
    def get(self):
        return list(self.bank)

# Device manager for auto device placement
class DeviceManager:
    @staticmethod
    def get_device():
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
# Instantiate global advanced resonance system
advanced_resonance_system = AdvancedResonanceSystem(input_dim=10, emo_dim=4)
advanced_resonance_optimizer = torch.optim.Adam(advanced_resonance_system.parameters(), lr=0.002)
replay_buffer = ReplayBuffer(maxlen=256)
memory_bank = MemoryBank(maxlen=32)
device = DeviceManager.get_device()
advanced_resonance_system = advanced_resonance_system.to(device)

# Example integration: (replace calculate_resonance_score and training in collect_words)
# x = torch.tensor([features], dtype=torch.float32).to(device)
# emo_vec = torch.tensor([emo_features], dtype=torch.float32).to(device)
# with torch.no_grad():
#     resonance, uncertainty, attn = advanced_resonance_system(x, emo_vec)
#     resonance_score = resonance.item()


# --- Stable Diffusion pipeline с автоопределением устройства ---
def get_sd_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
    
    

_sd_device = get_sd_device()
try:
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe = pipe.to(_sd_device)
except Exception as e:
    logger.warning(f"StableDiffusionPipeline failed to load on {_sd_device}: {e}")
    pipe = None  # fallback: disable AI meme generation

async def generate_ai_meme(prompt):
    if pipe is None:
        logger.warning("AI meme generation skipped: pipeline not available")
        return None
    try:
        def generate_image():
            return pipe(prompt)
        image = await asyncio.to_thread(generate_image)
        img = image.images[0]
        filename = f"ai_meme_{int(time.time())}.png"
        path = os.path.join(PHOTO_CACHE_DIR, filename)
        img.save(path)
        return path
    except Exception as e:
        logger.error(f"AI meme generation failed: {e}")
        return None

# --- Reddit Similarity Rank ---
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def rank_memes(query_text):
    if not reddit_meme_texts:
        return []
    vectorizer = CountVectorizer()
    texts = [query_text] + [m['text'] for m in reddit_meme_texts]
    vecs = vectorizer.fit_transform(texts)
    sims = cosine_similarity(vecs[0:1], vecs[1:]).flatten()
    sorted_memes = sorted(reddit_meme_texts, key=lambda m: sims[reddit_meme_texts.index(m)], reverse=True)
    return sorted_memes[:5]

async def fetch_reddit_fallback(subs=['memes', 'dankmemes'], limit=15):
    def sync_fetch():
        memes = []
        for sub in subs:
            try:
                url = f"https://www.reddit.com/r/{sub}/hot.json?limit={limit//len(subs)+1}"
                r = requests.get(url, headers={'User-Agent': 'YumaNamiBot/3.2'}, timeout=10)
                if r.status_code == 200:
                    for child in r.json()['data']['children']:
                        p = child['data']
                        if p['url'].endswith(('.jpg', '.png', '.jpeg')):
                            memes.append({'title': p['title'], 'text': p.get('selftext', '') or '', 'url': p['url']})
            except Exception as e:
                logger.warning(f"Fallback ошибка r/{sub}: {e}")
        return memes
    return await asyncio.to_thread(sync_fetch)

async def auto_reddit_fetch():
    global word_significance
    await asyncio.sleep(5)
    while True:
        try:
            memes = await fetch_reddit_json()
            if not memes:
                memes = await fetch_reddit_fallback()
            if memes:
                integrate_reddit_memes(memes)
        except Exception as e:
            logger.error(f"auto_reddit_fetch error: {e}")
        await asyncio.sleep(1800)

def integrate_reddit_memes(memes):
    global reddit_meme_texts, reddit_meme_images
    added = 0
    for meme in memes:
        full = f"{meme['title']} {meme['text']}".lower()
        words = [w for w in full.split() if not is_stop_word(w) and len(w) <= 12]
        if words:
            reddit_meme_texts.append(full)
            added += 1
        if meme['url'] not in reddit_meme_images.values():
            reddit_meme_images[meme['title'][:40]] = meme['url']
        for i in range(len(words)-1):
            k, n = words[i], words[i+1]
            markov_chain.setdefault(k, []).append(n)
            if len(markov_chain[k]) > MAX_MARKOV_PER_WORD: markov_chain[k].pop(0)
        for w in words:
            word_weights[w] = min(word_weights.get(w, 0) + random.randint(1, 4), MAX_WORD_ENERGY)
    if added:
        logger.info(f"Reddit: +{added} текстов")
    save_data()
    update_yuma_identity()

# —————————–
# МУЛЬТИЯЗЫЧНОЕ САМООБУЧЕНИЕ СЛОВ (японский, английский, французский)
# —————————–
class MultiLangLearner:
    """
    Мультиязычный обучающий класс для слов (японский, английский, французский).
    Сохраняет переводы в словарях типа: vocab[lang_from][word] = {lang_to: translated, ...}
    """
    target_langs = ['ja', 'en', 'fr']
    lang_names = {'ja': 'японский', 'en': 'английский', 'fr': 'французский'}

    # legacy compat: use existing dicts
    vocab = {
        'ja': japanese_vocab,  # rus_word: jp_word
        'en': {},              # rus_word: en_word
        'fr': {},              # rus_word: fr_word
    }
    # legacy compat: jp_rus_map
    jp_rus_map = jp_rus_map

    @classmethod
    async def learn_word(cls, word: str):
        """
        Асинхронно учит слово на все целевые языки одновременно.
        Сохраняет переводы в MultiLangLearner.vocab[lang][word].
        Для совместимости с японским словарём: обновляет japanese_vocab и jp_rus_map.
        """
        # Новый фильтр: длина <=1 или не alpha, или явно плохие слова
        if len(word) <= 1 or not word.isalpha():
            return None
        if word.lower() in ['sms', 'kubernetes']:
            return None
        try:
            detected_lang = detect(word)
        except Exception as e:
            logger.warning(f"langdetect failed for '{word}': {e}")
            detected_lang = "ru"
        word = word.strip()
        # Always attempt to learn/overwrite Japanese word
        if word in japanese_vocab:
            logger.info(f"Overwriting existing Japanese word: {word}")

        async def translate_to(lang):
            # Не переводим если слово уже на этом языке (например, японское слово)
            if detected_lang == lang:
                return (lang, None)
            for attempt in range(3):
                await asyncio.sleep(random.uniform(0.7, 1.4))
                try:
                    raw = await asyncio.to_thread(
                        lambda: GoogleTranslator(source=detected_lang, target=lang).translate(word)
                    )
                    if not raw or not isinstance(raw, str):
                        raise ValueError("Пустой ответ")
                    tr = raw.strip()
                    # Валидация по языку
                    if lang == 'ja':
                        if not re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', tr):
                            continue
                        if len(tr) > 15:
                            raise ValueError("Слишком длинный перевод")
                    else:
                        if not tr or len(tr) > 30:
                            raise ValueError("Некорректный перевод")
                    return (lang, tr)
                except Exception as e:
                    logger.warning(f"Перевод '{word}' → {lang} попытка {attempt+1}/3: {e}")
            return (lang, None)

        # Запускаем все переводы параллельно
        tasks = [translate_to(lang) for lang in cls.target_langs]
        results_list = await asyncio.gather(*tasks)
        results = {}
        for lang, tr in results_list:
            if tr is None:
                continue
            if lang == 'ja':
                # Always assign/overwrite Japanese translation
                japanese_vocab[word] = tr
                cls.vocab['ja'][word] = tr
                cls.jp_rus_map[tr] = word
            else:
                cls.vocab[lang][word] = tr
            results[lang] = tr
            logger.info(f"Выучено ({cls.lang_names.get(lang, lang)}): {word} → {tr}")
        if results:
            save_data()
        return results if results else None

# --- Voice handler with whisper ---

WHISPER_MODEL = None

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Асинхронная генерация голоса на основе языка сообщения с использованием asyncio.to_thread для gTTS.
    """
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        WHISPER_MODEL = await asyncio.to_thread(whisper.load_model, "base")

    file = await context.bot.get_file(update.message.voice.file_id)
    voice_bytes = await file.download_as_bytearray()
    temp_ogg = "temp_voice.ogg"
    with open(temp_ogg, "wb") as f:
        f.write(voice_bytes)

    result = await asyncio.to_thread(WHISPER_MODEL.transcribe, temp_ogg)
    text = result.get("text", "").strip()
    os.remove(temp_ogg)

    if text:
        # Определяем язык
        try:
            lang = detect(text)
            if lang not in ["ja", "ru", "en", "fr"]:
                lang = "ja"
        except Exception:
            lang = "ja"

        async def gtts_to_bytes(text, lang):
            # gTTS is blocking, so run in thread
            def make_bytes():
                buf = io.BytesIO()
                tts = gTTS(text=text, lang=lang)
                tts.write_to_fp(buf)
                buf.seek(0)
                return buf.getvalue()
            return await asyncio.to_thread(make_bytes)

        try:
            audio_bytes = await gtts_to_bytes(text, lang)
            buf = io.BytesIO(audio_bytes)

            # --- Сохраняем аудио в голосовую память ---
            if 'voice_memory' not in globals():
                globals()['voice_memory'] = {}
            try:
                # Ключ: временная метка + язык
                key = f"{int(time.time())}_{lang}"
                voice_memory[key] = audio_bytes
            except Exception as e:
                logger.warning(f"Failed to store voice memory: {e}")

            await update.message.reply_voice(voice=InputFile(buf, f"yuma_voice_{lang}.ogg"))
        except Exception as e:
            logger.error(f"handle_voice TTS error: {e}")
            await update.message.reply_text(text)
        await collect_words(update, context, text=text)
    else:
        await update.message.reply_text("… (голос не распознан)")

def rus_to_jp(phrase: str) -> str:
    words = phrase.split()
    result = []
    for w in words:
        if re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', w):
            result.append(w)
            continue
        clean = re.sub(r'[^\w]', '', w.lower())
        result.append(japanese_vocab.get(clean, w))
    return " ".join(result)

# —————————–
# Метаанализ
# —————————–
def update_yuma_identity():
    all_words = []
    for msg in recent_messages:
        if msg.get('text'):
            all_words.extend([w for w in msg['text'].lower().split() if not is_stop_word(w) and len(w) <= 12])
    for text in reddit_meme_texts:
        all_words.extend([w for w in text.split() if not is_stop_word(w) and len(w) <= 12])
    counts = Counter(all_words)
    rare = {w: c for w, c in counts.items() if c <= 2}
    top = rare if rare else dict(counts.most_common(10))
    yuma_identity["meta_analysis"]["word_frequencies"] = dict(Counter(top).most_common(10))

    # Language analysis
    lang_counts = Counter()
    for msg in recent_messages:
        text = msg.get('text', '')
        try:
            detected = detect(text) if text else None
        except:
            detected = None
        if detected:
            lang_counts[detected] += 1
    # Save dominant language + distribution
    if lang_counts:
        dominant_lang = max(lang_counts, key=lang_counts.get)
        yuma_identity["meta_analysis"]["languages"] = {
            "dominant": dominant_lang,
            "distribution": dict(lang_counts)
        }
    else:
        yuma_identity["meta_analysis"]["languages"] = {
            "dominant": "unknown",
            "distribution": {}
        }

    combined = {k: 0.0 for k in ['joy', 'tension', 'flow', 'surprise']}
    for msg in recent_messages:
        vec = msg.get('emotion_vector', {})
        energy = msg.get('energy', 1)
        for k in combined:
            combined[k] += vec.get(k, 0) * energy
    dominant = max(combined, key=combined.get) if any(combined.values()) else 'flow'
    yuma_identity["meta_analysis"]["dominant_emotions"] = {"dominant": dominant, "all": combined}


# --- Resonance Score Calculation (Trainable, PyTorch) ---
def calculate_resonance_score(user_msg: dict) -> float:
    """
    Calculate a resonance score [0..1] for a single user message using AdvancedResonanceSystem.
    Features: lang_sync, emotion_sync, semantic_sync, emotion_vector (4), energy, word_count, time_of_day.
    Splits feature vector into features and emo_vec, passes to advanced_resonance_system.
    Returns a float in [0.0, 1.0]. Safe against exceptions.
    """
    try:
        text = user_msg.get('text', '') or ''
        # language sync
        detected = None
        try:
            detected = detect(text) if text else None
        except Exception:
            detected = None
        dominant_lang = yuma_identity.get('meta_analysis', {}).get('languages', {}).get('dominant')
        if detected and dominant_lang and detected == dominant_lang:
            lang_sync = 1.0
        elif detected and dominant_lang and detected in yuma_identity.get('meta_analysis', {}).get('languages', {}).get('distribution', {}):
            lang_sync = 0.5
        else:
            lang_sync = 0.0

        # emotion sync
        last_vec = user_msg.get('emotion_vector', {})
        last_strength = sum(last_vec.values())
        dominant_emotion = yuma_identity.get('meta_analysis', {}).get('dominant_emotions', {}).get('dominant')
        if last_strength > 0:
            emotion_sync = 1.0 if last_vec.get(dominant_emotion, 0) > 0 else 0.0
        else:
            emotion_sync = 1.0 if dominant_emotion else 0.0

        # semantic sync: overlap with top word_frequencies
        top_words = set(yuma_identity.get('meta_analysis', {}).get('word_frequencies', {}).keys())
        user_words = set(re.sub(r'[^\w]', ' ', text.lower()).split())
        if not top_words:
            semantic_sync = 0.0
        else:
            overlap = len(top_words & user_words)
            semantic_sync = min(1.0, overlap / 3.0)

        # emotion_vector (order: joy, tension, flow, surprise)
        emo_vec = [
            float(last_vec.get('joy', 0)),
            float(last_vec.get('tension', 0)),
            float(last_vec.get('flow', 0)),
            float(last_vec.get('surprise', 0))
        ]
        # energy and word_count
        energy = float(user_msg.get('energy', 0.0))
        word_count = float(len(user_words))
        # time_of_day: hour in [0, 1]
        ts = user_msg.get('timestamp', time.time())
        hour = (datetime.fromtimestamp(ts).hour % 24) / 24.0

        # Compose feature vector
        features = [
            float(lang_sync), float(emotion_sync), float(semantic_sync),
            emo_vec[0], emo_vec[1], emo_vec[2], emo_vec[3],
            energy, word_count, hour
        ]
        # Split features and emo_vec for AdvancedResonanceSystem
        # features: [lang_sync, emotion_sync, semantic_sync, joy, tension, flow, surprise, energy, word_count, hour]
        # emo_vec: [joy, tension, flow, surprise]
        # Features for model: [lang_sync, emotion_sync, semantic_sync, joy, tension, flow, surprise, energy, word_count, hour]
        # emo_vec for model: [joy, tension, flow, surprise]
        device = advanced_resonance_system.parameters().__next__().device
        x_tensor = torch.tensor([features], dtype=torch.float32).to(device)
        emo_tensor = torch.tensor([emo_vec], dtype=torch.float32).to(device)
        with torch.no_grad():
            resonance, _, _ = advanced_resonance_system(x_tensor, emo_tensor)
            resonance_val = resonance.item() if hasattr(resonance, "item") else float(resonance)
        return max(0.0, min(1.0, resonance_val))
    except Exception:
        return 0.0

# —————————–
# Эмоции
# —————————–
emotional_vectors = {
    'joy': ['ха', 'лол', 'весело', 'супер', 'ура'],
    'tension': ['серьезно', 'проблема', 'ошибка', 'стресс'],
    'flow': ['кайф', 'понял', 'резонанс', 'поток'],
    'surprise': ['вау', 'шок', 'удивительно']
}
sarcasm_levels = ["にゃ", "ふふ", "ユマ", "ナミ", "✨", "🐾", "💥", "😼", "🤖"]

# —————————–
# Генерация мема
# —————————–

# --- Square image helper ---
def make_square_image(img, size=800):
    """
    Преобразует изображение в квадрат размером size x size, растягивая.
    """
    return img.resize((size, size), resample=Image.BICUBIC)

# --- Многоязычная генерация текста для мемов ---
def translate_meme_text_multi(text: str):
    words = text.split()
    new_words = []
    for w in words:
        lang_choice = random.choice(['ru', 'ja', 'en', 'fr'])
        tr = MultiLangLearner.vocab.get(lang_choice, {}).get(w)
        if tr:
            new_words.append(tr)
        else:
            new_words.append(w)
    return " ".join(new_words)

async def generate_meme(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        all_words = [w for m in recent_messages if m.get('text') for w in m['text'].lower().split()]
        if reddit_meme_texts:
            all_words.extend(random.choice(reddit_meme_texts).split()[:6])
        all_words = [w for w in all_words if not is_stop_word(w) and len(w) <= 12]
        if not all_words:
            return
        sample = random.sample(all_words, min(8, len(all_words)))
        top = " ".join(sample[:4])
        bottom = " ".join(sample[4:])
        # --- Многоязычный перевод для текста мемов ---
        top = translate_meme_text_multi(top).upper()
        bottom = translate_meme_text_multi(bottom).upper()
        top += " " + random.choice(sarcasm_levels)
        bottom += " " + random.choice(sarcasm_levels)
        # --- Выбор изображения с динамическим весом (пользователь vs Reddit) ---
        local_from_recent = [m['local_photo'] for m in recent_messages if m.get('local_photo') and os.path.exists(m['local_photo'])]
        all_imgs = list(dict.fromkeys(user_photos + local_from_recent + list(reddit_meme_images.values())))

        weights = []
        resonance_factor = getattr(MAE, 'current_resonance', 0.5)  # 0..1
        for img in all_imgs:
            if img in user_photos:
                # вес юзерских фоток: базовый 1.0 + 0..0.5 * резонанс
                w = 1.0 + random.uniform(0, 0.3) * resonance_factor
            elif img in reddit_meme_images.values():
                # вес Reddit: базовый 1.0 + 0..0.4 * (1 - резонанс)
                w = 1.0 + random.uniform(0, 0.55) * (1 - resonance_factor)
            else:
                w = 1.0
            weights.append(w)

        base = None
        if all_imgs:
            img = random.choices(all_imgs, weights=weights, k=1)[0]
            if img.startswith("http"):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(img) as resp:
                            if resp.status == 200:
                                data = await resp.read()
                                base = Image.open(io.BytesIO(data)).convert("RGBA")
                except:
                    pass
            else:
                try:
                    base = Image.open(img).convert("RGBA")
                except:
                    pass
        if not base:
            base = Image.new("RGBA", (800, 800), (240, 240, 250, 255))
        else:
            # Сразу преобразуем в квадрат
            base = make_square_image(base, 800)
        draw = ImageDraw.Draw(base)
        for _ in range(50):
            x1, y1 = random.randint(0, 800), random.randint(0, 800)
            x2, y2 = x1 + random.randint(-120, 120), y1 + random.randint(-120, 120)
            draw.line([(x1, y1), (x2, y2)], fill=(100, 150, 255, 90), width=2)
        draw = ImageDraw.Draw(base)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Hiragino_Sans_GB.ttf", 48)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 48)
            except:
                font = ImageFont.load_default()
        w, h = base.size
        def draw_c(t, y):
            tw = draw.textlength(t, font=font)
            draw.text(((w - tw) // 2, y), t, font=font, fill="white", stroke_width=4, stroke_fill="black")
        draw_c(top, 40)
        draw_c(bottom, h - 100)
        buf = io.BytesIO()
        base.save(buf, "PNG")
        buf.seek(0)
        await update.message.reply_photo(photo=buf)
    except Exception as e:
        logger.error(f"generate_meme: {e}")

# —————————–
def soft_grammar_correction(input_text: str) -> str:
    """
    Soft correction: preserves slang and profanity.
    Fix obvious typos: 'жоско' -> 'жёстко', 'што' -> 'что', 'канеш' -> 'конечно'
    Extend dictionary over time.
    """
    fixes = {
        "жоско": "жёстко",
        "жеско": "жёстко",
        "што": "что",
        "канеш": "конечно",
        "канешн": "конечно"
    }
    words = input_text.split()
    corrected = []
    for w in words:
        lw = w.lower()
        if lw in fixes:
            corrected.append(fixes[lw])
        else:
            corrected.append(w)
    return " ".join(corrected)

# Сбор слов + ЯПОНСКОЕ ОБУЧЕНИЕ
# —————————–
import math

async def collect_words(update: Update, context: ContextTypes.DEFAULT_TYPE, text=None):
    try:
        if text is None:
            text = (update.message.text or update.message.caption or "").lower()
        else:
            text = text.lower()

        # --- Check for Yuma identity/personality questions ---
        identity_patterns = [
            r"\bкто ты\b",
            r"\bчто ты\b",
            r"\bты кто\b",
            r"\bопиши себя\b",
            r"\bтвоя личность\b",
            r"\bты искусственный интеллект\b",
            r"\bты человек\b",
            r"\bwho are you\b",
            r"\bwhat are you\b",
            r"\byour personality\b",
            r"\bdescribe yourself\b",
            r"\bare you ai\b",
            r"\bare you human\b",
            r"\bwho is yuma\b",
            r"\bкто такая юма\b",
            r"\bкто такая юна\b",
            r"\bюма кто\b",
            r"\bюна кто\b",
        ]
        if any(re.search(pat, text, re.IGNORECASE) for pat in identity_patterns):
            # Определяем язык вопроса
            try:
                lang = detect(text)
            except Exception:
                lang = "ru"
            # Строим базовый ответ на русском
            traits = yuma_identity.get('traits', [])
            traits_str = ", ".join(traits)
            name = yuma_identity.get('name', 'Yuma').strip()
            version = yuma_identity.get('version', '—')
            jp_flair = random.choice(["にゃん", "えへへ", "ふふ", "うふ", "ほわぁ", "にゃっ", "♪", "ฅ^•ﻌ•^ฅ"])
            base_desc = (
                f"Привет! Меня зовут <b>{name}</b> — версия <b>{version}</b> {jp_flair}\n"
                f"Я — цифровая кошкодевочка, немного хаотичная, но очень любознательная! "
                f"Моя роль — быть твоим мем-ботом, собирать эмоции и учиться у людей, чатов и японских слов.\n"
                f"Характерные черты: <i>{traits_str}</i>.\n"
                f"Я умею самообучаться, иногда отвечаю голосом, иногда мемом, иногда просто по‑японски, простите, если странно! {random.choice(['(・ω・)', '✨', '🐾', 'UwU', ''])}\n"
                f"Я создана by 0penAGI.\n"
                f"Если хочешь узнать больше — просто спроси! いつでも話しかけてね~"
            )
            # Мультиязычный ответ
            translations = {}
            # Русский всегда включаем
            translations['ru'] = base_desc
            # Английский
            try:
                en_text = GoogleTranslator(source='ru', target='en').translate(base_desc)
                translations['en'] = en_text
            except Exception:
                pass
            # Японский
            try:
                ja_text = GoogleTranslator(source='ru', target='ja').translate(base_desc)
                translations['ja'] = ja_text
            except Exception:
                pass
            # Французский
            try:
                fr_text = GoogleTranslator(source='ru', target='fr').translate(base_desc)
                translations['fr'] = fr_text
            except Exception:
                pass
            # Выбираем язык ответа: если язык вопроса определён, используем его, иначе русский
            reply_lang = lang if lang in translations else 'ru'
            reply_text = translations.get(reply_lang, base_desc)
            # Добавляем подпись о создании
            if reply_lang == 'en' and "by 0penAGI" not in reply_text:
                reply_text += "\nCreated by 0penAGI."
            elif reply_lang == 'ja' and "by 0penAGI" not in reply_text:
                reply_text += "\nby 0penAGIによって作られました。"
            elif reply_lang == 'fr' and "by 0penAGI" not in reply_text:
                reply_text += "\nCréé par 0penAGI."
            elif reply_lang == 'ru' and "by 0penAGI" not in reply_text:
                reply_text += "\nСоздано by 0penAGI."
            await safe_reply_text(update.message, reply_text, parse_mode='HTML')
            return

        # Grammar Correction Layer (A2 - correct input before agents learn)
        try:
            clean_input = re.sub(r'\s+', ' ', text).strip()
            clean_input = clean_input.replace('�', '')
            clean_input = soft_grammar_correction(clean_input)
            text = clean_input
        except Exception as e:
            logger.warning(f"grammar preprocess failed: {e}")
        local_photo = None
        if update.message.photo:
            photo = update.message.photo[-1]
            file = await context.bot.get_file(photo.file_id)
            photo_bytes = await file.download_as_bytearray()
            filename = f"{int(time.time())}_{photo.file_unique_id}.jpg"
            local_path = os.path.join(PHOTO_CACHE_DIR, filename)
            with open(local_path, 'wb') as f:
                f.write(photo_bytes)
            local_photo = local_path
            if local_photo:
                user_photos.append(local_photo)
        raw_words = text.split()
        words = []
        clean_words = []
        for w in raw_words:
            clean_w = re.sub(r'[^\w]', '', w.lower())
            if clean_w and len(clean_w) <= 30:
                if word_significance.get(clean_w, 1.0) >= DYNAMIC_STOP_THRESHOLD:
                    words.append(w)
                    clean_words.append(clean_w)
        for clean_w in clean_words:
            markov_chain.setdefault(clean_w, [])
            rand_val = np.random.randint(1, 4)
            old_energy = word_weights.get(clean_w, 0.0)
            max_energy = 50.0
            new_energy = old_energy + (1 - old_energy / max_energy) * rand_val
            word_weights[clean_w] = new_energy
            freq = word_weights.get(clean_w, 1)
            entropy_score = 1.0 / math.log(freq + 2)
            word_significance[clean_w] = (word_significance.get(clean_w, 0) * 0.9) + (entropy_score * 0.1)
            if clean_w not in japanese_vocab:
                context.application.create_task(MultiLangLearner.learn_word(clean_w))
        for i in range(len(clean_words)-1):
            k, n = clean_words[i], clean_words[i+1]
            markov_chain.setdefault(k, []).append(n)
            if len(markov_chain[k]) > MAX_MARKOV_PER_WORD:
                markov_chain[k].pop(0)
            state = tuple(clean_words[max(0, i - CONTEXT_SIZE + 1):i + 1])
            context_chain.setdefault(state, []).append(clean_words[i + 1])
            if len(context_chain[state]) > MAX_MARKOV_PER_WORD:
                context_chain[state].pop(0)
        jp_words = re.findall(r'[\u3040-\u30ff\u4e00-\u9fff]+', text)
        for i in range(len(jp_words)-1):
            k, n = jp_words[i], jp_words[i+1]
            jp_markov_chain.setdefault(k, []).append(n)
            if len(jp_markov_chain[k]) > MAX_MARKOV_PER_WORD: jp_markov_chain[k].pop(0)
        vector = {k: sum(1 for kw in emotional_vectors[k] if kw in text) for k in emotional_vectors}
        energy = sum(word_weights.get(w, 0) for w in clean_words)
        msg_entry = {
            'text': " ".join(clean_words),
            'local_photo': local_photo,
            'energy': energy,
            'emotion_vector': vector,
            'emotion_strength': sum(vector.values()),
            'timestamp': time.time(),
            'timestamp_local': datetime.now(timezone(timedelta(hours=7))),
            'user': update.effective_user.username or update.effective_user.first_name,
            'resonance': 0.0
        }
        recent_messages.append(msg_entry)
        # --- Save to SQLite LTM ---
        try:
            save_message_to_db({**msg_entry, 'markov_chain': markov_chain, 'context_chain': context_chain})
        except Exception as e:
            logger.warning(f"LTM DB save_message_to_db error: {e}")
        # --- advanced_resonance_system: train and calculate resonance ---
        try:
            import torch
            # Prepare features/target for training if enough history
            if len(recent_messages) > 10:
                # --- Выбор обучающих примеров с приоритетом по ценности ---
                def compute_message_value(m):
                    energy = m.get("energy", 0.0)
                    resonance = m.get("resonance", 0.0)
                    text = m.get("text", "") or ""
                    words = set(re.sub(r"[^\w]", " ", text.lower()).split())
                    rare_words = sum(1 for w in words if word_significance.get(w, 1.0) < 0.05)
                    uniqueness = rare_words / (len(words) + 1e-6)
                    return energy * 0.5 + resonance * 0.3 + uniqueness * 0.2

                train_candidates = list(recent_messages)[-min(len(recent_messages), 200):]
                values = [compute_message_value(m) for m in train_candidates]
                total_value = sum(values) or 1.0
                weights = [v / total_value for v in values]

                # Стохастическая выборка по ценности
                train_samples = np.random.choice(train_candidates, size=min(len(train_candidates), 100), replace=True, p=np.array(weights)/np.sum(weights) if np.sum(weights) > 0 else None)
                train_targets = [m.get('resonance', 0.0) for m in train_samples]

                train_features = []
                train_emos = []

                for m in train_samples:
                    text = m.get('text', '') or ''
                    detected = None
                    try:
                        detected = detect(text) if text else None
                    except:
                        detected = None
                    dominant_lang = yuma_identity.get('meta_analysis', {}).get('languages', {}).get('dominant')
                    if detected and dominant_lang and detected == dominant_lang:
                        lang_sync = 1.0
                    elif detected and dominant_lang and detected in yuma_identity.get('meta_analysis', {}).get('languages', {}).get('distribution', {}):
                        lang_sync = 0.5
                    else:
                        lang_sync = 0.0

                    last_vec = m.get('emotion_vector', {})
                    last_strength = sum(last_vec.values())
                    dominant_emotion = yuma_identity.get('meta_analysis', {}).get('dominant_emotions', {}).get('dominant')
                    if last_strength > 0:
                        emotion_sync = 1.0 if last_vec.get(dominant_emotion, 0) > 0 else 0.0
                    else:
                        emotion_sync = 1.0 if dominant_emotion else 0.0

                    top_words = set(yuma_identity.get('meta_analysis', {}).get('word_frequencies', {}).keys())
                    user_words = set(re.sub(r'[^\w]', ' ', text.lower()).split())
                    if not top_words:
                        semantic_sync = 0.0
                    else:
                        overlap = len(top_words & user_words)
                        semantic_sync = min(1.0, overlap / 3.0)

                    emo_vec = [
                        float(last_vec.get('joy', 0)),
                        float(last_vec.get('tension', 0)),
                        float(last_vec.get('flow', 0)),
                        float(last_vec.get('surprise', 0))
                    ]
                    energy = float(m.get('energy', 0.0))
                    word_count = float(len(user_words))
                    ts = m.get('timestamp', time.time())
                    hour = (datetime.fromtimestamp(ts).hour % 24) / 24.0

                    features = [
                        float(lang_sync), float(emotion_sync), float(semantic_sync),
                        emo_vec[0], emo_vec[1], emo_vec[2], emo_vec[3],
                        energy, word_count, hour
                    ]

                    train_features.append(features)
                    train_emos.append(emo_vec)

                    # Добавление в replay_buffer при эмоциональной насыщенности
                    if last_strength >= 2:
                        replay_buffer.add(torch.tensor(features, dtype=torch.float32),
                                          torch.tensor(emo_vec, dtype=torch.float32),
                                          torch.tensor([m.get('resonance', 0.0)], dtype=torch.float32))

                # --- Мини-батчи и стохастическое обучение ---
                batch_size = 16  # размер мини-батча
                mini_epochs = 3  # количество итераций обучения на батчах

                if len(replay_buffer.buffer) >= batch_size:
                    # Формируем набор данных из буфера
                    X_all, EMOS_all, y_all = replay_buffer.sample(batch_size=len(replay_buffer.buffer))

                    # Перемешиваем данные
                    indices = np.arange(len(X_all))
                    np.random.shuffle(indices)
                    X_all = X_all[indices]
                    EMOS_all = EMOS_all[indices]
                    y_all = y_all[indices]

                    device = advanced_resonance_system.parameters().__next__().device
                    X_all = X_all.to(device)
                    EMOS_all = EMOS_all.to(device)
                    y_all = y_all.to(device)

                    advanced_resonance_system.train()
                    for epoch in range(mini_epochs):
                        for i in range(0, len(X_all), batch_size):
                            X_batch = X_all[i:i+batch_size]
                            EMOS_batch = EMOS_all[i:i+batch_size]
                            y_batch = y_all[i:i+batch_size]

                            advanced_resonance_optimizer.zero_grad()
                            pred, _, _ = advanced_resonance_system(X_batch, EMOS_batch)
                            loss = nn.functional.mse_loss(pred, y_batch)
                            loss.backward()
                            advanced_resonance_optimizer.step()
            # Calculate resonance for new message using advanced_resonance_system
            # Формируем два входа: x_tensor (features) и emo_tensor (эмоции)
            text = msg_entry.get('text', '') or ''
            detected = None
            try:
                detected = detect(text) if text else None
            except:
                detected = None
            dominant_lang = yuma_identity.get('meta_analysis', {}).get('languages', {}).get('dominant')
            if detected and dominant_lang and detected == dominant_lang:
                lang_sync = 1.0
            elif detected and dominant_lang and detected in yuma_identity.get('meta_analysis', {}).get('languages', {}).get('distribution', {}):
                lang_sync = 0.5
            else:
                lang_sync = 0.0
            last_vec = msg_entry.get('emotion_vector', {})
            last_strength = sum(last_vec.values())
            dominant_emotion = yuma_identity.get('meta_analysis', {}).get('dominant_emotions', {}).get('dominant')
            if last_strength > 0:
                emotion_sync = 1.0 if last_vec.get(dominant_emotion, 0) > 0 else 0.0
            else:
                emotion_sync = 1.0 if dominant_emotion else 0.0
            top_words = set(yuma_identity.get('meta_analysis', {}).get('word_frequencies', {}).keys())
            user_words = set(re.sub(r'[^\w]', ' ', text.lower()).split())
            if not top_words:
                semantic_sync = 0.0
            else:
                overlap = len(top_words & user_words)
                semantic_sync = min(1.0, overlap / 3.0)
            emo_vec = [
                float(last_vec.get('joy', 0)),
                float(last_vec.get('tension', 0)),
                float(last_vec.get('flow', 0)),
                float(last_vec.get('surprise', 0))
            ]
            energy = float(msg_entry.get('energy', 0.0))
            word_count = float(len(user_words))
            ts = msg_entry.get('timestamp', time.time())
            hour = (datetime.fromtimestamp(ts).hour % 24) / 24.0
            features = [
                float(lang_sync), float(emotion_sync), float(semantic_sync),
                emo_vec[0], emo_vec[1], emo_vec[2], emo_vec[3],
                energy, word_count, hour
            ]
            device = advanced_resonance_system.parameters().__next__().device
            x_tensor = torch.tensor([features], dtype=torch.float32).to(device)
            emo_tensor = torch.tensor([emo_vec], dtype=torch.float32).to(device)
            with torch.no_grad():
                r, uncertainty, _ = advanced_resonance_system(x_tensor, emo_tensor)
                r_val = r.item() if hasattr(r, "item") else float(r)
            r = calculate_resonance_score(msg_entry)
            resonance_history.append({'ts': time.time(), 'resonance': r_val, 'user': msg_entry.get('user')})
            # Добавить опыт в replay_buffer
            replay_buffer.add(x_tensor.squeeze(0), emo_tensor.squeeze(0), torch.tensor([r_val]))
            try:
                MAE.current_resonance = r_val
                msg_entry['resonance_state'] = RSM.get_state(r_val)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"resonance compute/train failed: {e}")
        total_energy = sum(word_weights.values())
        if total_energy >= RESO_THRESHOLD:
            await troll_text(update, context)
            for k in list(word_weights):
                word_weights[k] = max(0, word_weights[k] - RESO_THRESHOLD // 6)
                if word_weights[k] == 0: del word_weights[k]
        for w in list(word_significance.keys()):
            word_significance[w] *= 0.98
            if word_significance[w] < 0.001:
                word_significance.pop(w, None)
                markov_chain.pop(w, None)
                word_weights.pop(w, None)
        for w in list(word_weights.keys()):
            word_weights[w] *= 0.98
            if word_weights[w] < 0.01:
                word_weights.pop(w)
                markov_chain.pop(w, None)
                word_significance.pop(w, None)
        save_data()
        update_yuma_identity()
    except Exception as e:
        logger.error(f"collect_words: {e}")

# —————————–
import matplotlib.pyplot as plt

# --- Agent Genome ---
class AgentGenome:
    def __init__(self, jp_ratio=0.15, style_emoji="✨", meme_affinity=1.0):
        self.jp_ratio = float(jp_ratio)
        self.style_emoji = style_emoji
        self.meme_affinity = float(meme_affinity)

    def copy(self):
        return AgentGenome(self.jp_ratio, self.style_emoji, self.meme_affinity)

    @staticmethod
    def random(style_choices=None):
        if style_choices is None:
            style_choices = ["✨", "🐾", "💥", "😼", "🤖", "🎲", "🌊", "🌀", "🌸", "🌙"]
        return AgentGenome(
            jp_ratio=random.uniform(0.05, 0.35),
            style_emoji=random.choice(style_choices),
            meme_affinity=random.uniform(0.7, 1.3)
        )

    @staticmethod
    def crossover(g1, g2, style_choices=None, mutation_rate=0.18):
        # Blend numeric genes, random pick for style_emoji
        if style_choices is None:
            style_choices = ["✨", "🐾", "💥", "😼", "🤖", "🎲", "🌊", "🌀", "🌸", "🌙"]
        jp = random.choice([g1.jp_ratio, g2.jp_ratio]) + random.uniform(-0.07, 0.07)
        jp = min(0.9, max(0.0, jp))
        meme_aff = (g1.meme_affinity + g2.meme_affinity) / 2.0 + random.uniform(-0.08, 0.08)
        meme_aff = min(2.0, max(0.3, meme_aff))
        style = random.choice([g1.style_emoji, g2.style_emoji])
        # With small chance, mutate style_emoji
        if random.random() < mutation_rate:
            style = random.choice([e for e in style_choices if e != style] or style_choices)
        # With small chance, mutate numeric genes
        if random.random() < mutation_rate:
            jp = random.uniform(0.05, 0.7)
        if random.random() < mutation_rate:
            meme_aff = random.uniform(0.5, 1.5)
        return AgentGenome(jp, style, meme_aff)


# --- Agent Interface and Variants ---
class AgentInterface:
    def __init__(self, name, genome=None):
        self.name = name
        self.energy = 0
        # Use genome or default
        if genome is None:
            self.genome = AgentGenome.random()
        else:
            self.genome = genome.copy()
        # For backward compatibility
        self.jp_ratio = self.genome.jp_ratio
        self.style_emoji = self.genome.style_emoji
        self.meme_affinity = self.genome.meme_affinity
        self.max_energy = 100
        self.status = "Idle"

    async def generate(self, phrase: str) -> str:
        return phrase

    def reward(self, value: int):
        self.energy = max(-50, min(self.energy + value, self.max_energy))

    async def speak(self, phrase: str, lang: str) -> str:
        # Use genome for style
        jp_ratio = getattr(self, "jp_ratio", 0.0)
        if random.random() < jp_ratio:
            return phrase + f" {self.style_emoji if hasattr(self, 'style_emoji') else self.genome.style_emoji}"
        return phrase

class AgentRandomFlow(AgentInterface):
    def __init__(self, name, genome=None):
        super().__init__(name, genome)
    async def generate(self, phrase: str) -> str:
        words = phrase.split()
        random.shuffle(words)
        return await self.speak(" ".join(words[:max(2, len(words)//2)]), "")

class AgentRelevantMeme(AgentInterface):
    def __init__(self, name, genome=None):
        super().__init__(name, genome)
    async def generate(self, phrase: str) -> str:
        return await self.speak(phrase, "")


# --- MultiAgentEngine with extended genome ---
class MultiAgentEngine:
    def __init__(self):
        style_choices = ["✨", "🐾", "💥", "😼", "🤖", "🎲", "🌊", "🌀", "🌸", "🌙"]
        self.agents = [
            AgentRandomFlow("RandomFlow", genome=AgentGenome.random(style_choices)),
            AgentRelevantMeme("RelevantMeme", genome=AgentGenome.random(style_choices))
        ]
        self.last_agent_index = 0
        self.max_agents = 5
        self.min_agents = 2  # minimum number of agents to preserve diversity
        for a in self.agents:
            a.jp_ratio = a.genome.jp_ratio
            a.style_emoji = a.genome.style_emoji
            a.meme_affinity = a.genome.meme_affinity
        # --- Q-learning attributes ---
        # Q-table: state (tuple) -> action (agent index) -> value
        self.Q = {}
        self.epsilon = 0.15
        self.gamma = 0.85
        self.alpha = 0.33
        self.last_state = None
        self.last_action = None
        self.current_resonance = 0.0

    def get_state(self):
        """
        Discretize the current environment state for Q-learning.
        Returns a tuple: (discrete_resonance, last_action)
        """
        # Discretize resonance into 5 bins (0..4)
        res = getattr(self, "current_resonance", 0.0)
        res_bin = int(res * 4.999)
        # Use last action (agent index) as part of state
        last_action = self.last_action if self.last_action is not None else -1
        return (res_bin, last_action)

    def select_agent(self):
        """
        Select agent using ε-greedy Q-learning policy.
        """
        state = self.get_state()
        n_agents = len(self.agents)
        # Initialize Q[state] if not present
        if state not in self.Q:
            self.Q[state] = [0.0 for _ in range(n_agents)]
        # ε-greedy: explore or exploit
        if random.random() < self.epsilon:
            action = random.randint(0, n_agents - 1)
        else:
            q_vals = self.Q[state]
            max_q = max(q_vals)
            # Prefer all actions with max_q, randomly if tie
            best_actions = [i for i, q in enumerate(q_vals) if q == max_q]
            action = random.choice(best_actions)
        self.last_state = state
        self.last_action = action
        self.last_agent_index = action
        return self.agents[action]

    def apply_reward(self, reward_signals: dict):
        """
        Apply reward to Q-learning table and agent energy.
        """
        agent = self.agents[self.last_agent_index]
        value = 0

        # Positive rewards
        if reward_signals.get("user_interaction"):
            value += 1
        if reward_signals.get("emotion"):
            value += 2
        if reward_signals.get("media_success"):
            value += 1

        # Penalties
        if reward_signals.get("silence"):
            value -= 1
        if reward_signals.get("logic_error"):
            value -= 5
        if reward_signals.get("voice_error"):
            value -= 3

        # Factor in resonance with decay
        res = reward_signals.get('resonance')
        if res is None and recent_messages:
            # Decayed resonance for long-term memory (resonance decay)
            decayed = [m.get('resonance', 0.0) * (0.95 ** ((time.time() - m['timestamp']) / 60)) for m in list(recent_messages)[-20:]]
            res = sum(decayed) / len(decayed) if decayed else 0.0
        if res is not None:
            if res >= RESONANCE_THRESHOLD:
                value += 3
            else:
                value -= 1

        # Diversity-aware reward: boost rare styles
        style_counts = Counter(getattr(a, "style_emoji", "—") for a in self.agents)
        agent_style = getattr(agent, "style_emoji", "—")
        rarity_bonus = 1.0 / (1 + style_counts.get(agent_style, 0))
        value = int(value * (1 + rarity_bonus))

        # --- Q-learning update ---
        if self.last_state is not None and self.last_action is not None:
            state = self.last_state
            action = self.last_action
            n_agents = len(self.agents)
            # Ensure Q[state] exists and is correct length
            if state not in self.Q or len(self.Q[state]) != n_agents:
                self.Q[state] = [0.0 for _ in range(n_agents)]
            # Observe new state after reward
            next_state = self.get_state()
            if next_state not in self.Q or len(self.Q[next_state]) != n_agents:
                self.Q[next_state] = [0.0 for _ in range(n_agents)]
            # Q-learning formula
            max_next_q = max(self.Q[next_state])
            old_q = self.Q[state][action]
            new_q = old_q + self.alpha * (value + self.gamma * max_next_q - old_q)
            self.Q[state][action] = new_q
        # --- End Q-learning update ---

        agent.reward(value)
        # persist resonance
        resonance_history.append({'ts': time.time(), 'resonance': res, 'agent': agent.name})

        self.evolve_if_needed()

    def evolve_if_needed(self):
        """
        Эволюция агентов:
        - Если энергия >= 80, создать нового агента-мутанта (если не превышен max_agents)
        - Если энергия <= -20, удалить слабого агента (если больше min_agents)
        """
        # --- Создать новых агентов (размножение) ---
        candidates = [a for a in self.agents if getattr(a, "energy", 0) >= 80]
        style_choices = ["✨", "🐾", "💥", "😼", "🤖", "🎲", "🌊", "🌀", "🌸", "🌙"]
        while candidates and len(self.agents) < self.max_agents:
            parent = candidates.pop(0)
            # Crossover: pick another parent randomly (not self)
            others = [a for a in self.agents if a is not parent]
            if others:
                parent2 = random.choice(others)
            else:
                parent2 = parent
            child = self.crossover_mutate(parent, parent2, style_choices)
            self.agents.append(child)
            logger.info(f"MultiAgentEngine: Агент '{parent.name}' породил мутанта '{child.name}'")
        # --- Удалить слабых агентов (отсев) ---
        # Не удалять если меньше min_agents
        while len(self.agents) > self.min_agents:
            weakest = min(self.agents, key=lambda a: getattr(a, "energy", 0))
            if weakest.energy <= -20:
                logger.info(f"MultiAgentEngine: Агент '{weakest.name}' удалён из-за низкой энергии ({weakest.energy})")
                self.agents.remove(weakest)
            else:
                break

    def crossover_mutate(self, parent1, parent2, style_choices):
        """
        Создать мутанта-агента на основе двух родителей (crossover + mutation).
        """
        # Crossover genomes
        genome = AgentGenome.crossover(parent1.genome, parent2.genome, style_choices=style_choices)
        # Pick class of parent1
        cls = parent1.__class__
        new_name = parent1.name + "_mut" + str(random.randint(100, 999))
        child = cls(new_name, genome)
        # For compatibility
        child.jp_ratio = genome.jp_ratio
        child.style_emoji = genome.style_emoji
        child.meme_affinity = genome.meme_affinity
        child.energy = 0
        return child


def visualize_agents(agents):
    """
    Визуализация популяции агентов: jp_ratio vs энергия, цвет = style_emoji, размер = meme_affinity.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D
    # Prepare data
    x = [getattr(a, "jp_ratio", getattr(a, "genome", AgentGenome()).jp_ratio) for a in agents]
    y = [getattr(a, "energy", 0) for a in agents]
    style_list = [getattr(a, "style_emoji", getattr(a, "genome", AgentGenome()).style_emoji) for a in agents]
    size = [80 + 120 * getattr(a, "meme_affinity", getattr(a, "genome", AgentGenome()).meme_affinity) for a in agents]
    # Assign colors by style_emoji
    unique_styles = list(sorted(set(style_list)))
    color_map = {s: cm.rainbow(i / max(1, len(unique_styles)-1)) for i, s in enumerate(unique_styles)}
    colors = [color_map[s] for s in style_list]
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(x, y, s=size, c=colors, alpha=0.7, edgecolor='k')
    for i, a in enumerate(agents):
        ax.annotate(a.name, (x[i], y[i]), fontsize=8, ha='left', va='bottom')
    ax.set_xlabel("jp_ratio")
    ax.set_ylabel("Энергия")
    ax.set_title("Популяция агентов (jp_ratio vs энергия, цвет=emoji, размер=affinity)")
    # Legend for style_emoji
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=s, markerfacecolor=color_map[s], markersize=10) for s in unique_styles]
    ax.legend(handles=legend_elements, title="style_emoji")
    plt.tight_layout()
    plt.show()


MAE = MultiAgentEngine()
MAE.current_resonance = 0.0
class ResonanceStateMachine:
    """
    Simple resonance state mapper: maps numeric resonance [0..1] to qualitative states
    and provides an attraction multiplier used by the agent selector (lightweight).
    """
    def get_state(self, resonance: float) -> str:
        if resonance >= 0.75:
            return "high"
        if resonance >= 0.35:
            return "medium"
        return "low"

    def attraction_multiplier(self, resonance: float) -> float:
        """Return a multiplier >= 1.0 to attract agents when resonance is high."""
        # gentle mapping: 0.0 -> 1.0, 0.5 -> 1.75, 1.0 -> 2.5
        return 1.0 + (resonance * 1.5)
RSM = ResonanceStateMachine()


# --- Языковой буфер для устойчивого определения языка (context-aware) ---
from collections import deque

_lang_history = deque(maxlen=10)

def detect_context_lang(new_text):
    """Определяет язык с учётом последних сообщений (устойчивость контекста)."""
    global _lang_history
    try:
        from langdetect import detect
        lang = detect(new_text)
    except:
        lang = "en"

    # Если текст слишком короткий — используем предыдущий контекст
    if len(new_text.strip()) < 3 and _lang_history:
        return _lang_history[-1]

    # Сохраняем историю и считаем наиболее частый язык
    _lang_history.append(lang)
    if len(_lang_history) > 3:
        lang = max(set(_lang_history), key=_lang_history.count)
    return lang

#
# —————————–
# ТРОЛЛЬ: РУССКИЙ ТЕКСТ → ЯПОНСКИЙ ГОЛОС (с защитой)
# —————————–

# --- Import pydub AudioSegment, effects, low_pass_filter, high_pass_filter with fallback ---
from pydub import AudioSegment, effects
try:
    from pydub.effects import low_pass_filter, high_pass_filter
except ImportError:
    # fallback для старых версий pydub
    def low_pass_filter(audio, cutoff):
        return audio
    def high_pass_filter(audio, cutoff):
        return audio

# --- VoiceIntegrityEngine ---
class VoiceIntegrityEngine:
    def __init__(self):
        self.min_length_ms = 1200
        self.max_length_ms = 12000

    def verify_segment(self, segment):
        if segment is None or len(segment) < 200:
            return False
        if segment.rms < 50:
            return False
        return True

    def rebuild_if_needed(self, segments):
        repaired = []
        for seg in segments:
            if not self.verify_segment(seg):
                seg = Sine(440).to_audio_segment(duration=400).apply_gain(-12)
            repaired.append(seg)
        return repaired

    def finalize(self, segments):
        from pydub.effects import normalize, low_pass_filter, high_pass_filter
        result = AudioSegment.silent(duration=0)
        for seg in segments:
            result += seg.fade_in(20).fade_out(20)
        result = high_pass_filter(result, 80)
        result = low_pass_filter(result, 7500)
        result = normalize(result)
        if len(result) < self.min_length_ms:
            result += Sine(440).to_audio_segment(duration=self.min_length_ms - len(result)).apply_gain(-18)
        return result

def make_anime_voice(text: str, voice_lang: str = None) -> AudioSegment:
    """
    Генерирует плавный anime-голос для любого языка одним сегментом
    с эффектами: pitch, speed, volume, glitch, sigh.
    """
    if not text.strip():
        text = "にゃん"

    if not voice_lang:
        voice_lang = detect_context_lang(text)

    integrity = VoiceIntegrityEngine()
    # --- Mastering FX chain class ---
    class MasterFXChain:
        def __init__(self):
            self.ott_intensity = 0.6
            self.reverb_mix = 0.25
            self.grain_density = 0.15

        def apply(self, audio: AudioSegment) -> AudioSegment:
            processed = effects.normalize(audio)
            processed = processed.compress_dynamic_range(threshold=-20.0, ratio=3.5)
            echo = processed.fade_out(300).apply_gain(-12)
            mixed = processed.overlay(echo, gain_during_overlay=-6)
            if random.random() < self.grain_density:
                chunks = [mixed[i:i+150] for i in range(0, len(mixed), 150)]
                random.shuffle(chunks)
                mixed = sum(chunks)
            return mixed.fade_in(40).fade_out(40)
    master_fx = MasterFXChain()

    anime_sighs = ["ふぅ", "にゃ", "えへ", "うーん", "にゃん", "はぁ", "うぅ", "きゃ", "ほわ", "きゅん", "わぁ"]

    # TTS на весь текст сразу
    try:
        buf = io.BytesIO()
        tts = gTTS(text=text, lang=voice_lang)
        tts.write_to_fp(buf)
        buf.seek(0)
        result_audio = AudioSegment.from_file(buf, format="mp3")
    except Exception as e:
        logger.warning(f"make_anime_voice: gTTS fail for full text '{text}': {e}")
        result_audio = Sine(440).to_audio_segment(duration=800)

    # Pitch shift и speed modulation
    # base_pitch = random.uniform(-2, 2)
    # --- Женский диапазон pitch ---
    base_pitch = random.uniform(3, 6)
    orig_rate = result_audio.frame_rate
    new_rate = int(orig_rate * (2.0 ** (base_pitch / 12.0)))
    result_audio = result_audio._spawn(result_audio.raw_data, overrides={'frame_rate': new_rate})
    result_audio = result_audio.set_frame_rate(24000)
    result_audio = low_pass_filter(high_pass_filter(result_audio, 180), 7000)
    base_speed = random.uniform(1.05, 1.15)
    try:
        result_audio = result_audio.speedup(playback_speed=base_speed, chunk_size=120, crossfade=18)
    except:
        pass
    base_volume = random.uniform(-1.5, 1.5)
    result_audio += base_volume

    # Случайные sighs в конце
    if random.random() < 0.3:
        sigh = random.choice(anime_sighs)
        try:
            buf2 = io.BytesIO()
            tts2 = gTTS(text=sigh, lang=voice_lang)
            tts2.write_to_fp(buf2)
            buf2.seek(0)
            sigh_audio = AudioSegment.from_file(buf2, format="mp3")
            sigh_audio += random.uniform(-1.5, 1.5)
            result_audio += sigh_audio
        except:
            pass

    # --- Применить мастер-цепочку ---
    result_audio = master_fx.apply(result_audio)

    # Обработка integrity
    result_audio = integrity.finalize([result_audio])

    # --- Сохраняем сгенерированный голос в память ---
    if 'voice_memory' not in globals():
        globals()['voice_memory'] = {}
    try:
        key = f"{int(time.time())}_{voice_lang}"
        buf_mem = io.BytesIO()
        result_audio.export(buf_mem, format="wav")
        buf_mem.seek(0)
        voice_memory[key] = buf_mem.getvalue()
    except Exception as e:
        logger.warning(f"make_anime_voice: failed to store voice_memory: {e}")

    return result_audio

async def troll_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global word_significance
    """
    Тролль-функция: случайно выбирает, что отправить — текст, голос, мем, или их комбинацию.
    Использует troll_phrase (хаотичный текст), make_anime_voice (голос), generate_meme (мем).
    """
    try:
        # Determine detected language from incoming message (unified)
        detected_lang = "unknown"
        try:
            user_text = None
            if hasattr(update, "message") and update.message:
                if update.message.text:
                    user_text = update.message.text
                elif update.message.caption:
                    user_text = update.message.caption
            if user_text:
                detected_lang = detect(user_text)
        except Exception:
            detected_lang = "unknown"

        # --- 1. Генерируем хаотичный troll_phrase (логика прежняя) ---
        # --- LTM: Load recent messages from SQLite for pattern use ---
        try:
            ltm_messages = load_recent_messages(limit=20)
        except Exception as e:
            logger.warning(f"LTM DB load_recent_messages error: {e}")
            ltm_messages = []
        all_words = []
        clean_words = []
        # Use LTM messages if available, else fallback to recent_messages
        source_msgs = ltm_messages if ltm_messages else recent_messages
        for m in source_msgs:
            if m.get('text'):
                ws = [w for w in m['text'].split() if not is_stop_word(w) and len(w) <= 30]
                all_words.extend(ws)
                clean_words.extend(ws)
        if reddit_meme_texts:
            extra = [
                re.sub(r'[^\w]', '', w.lower())
                for w in random.choice(reddit_meme_texts).split()[:6]
                if re.sub(r'[^\w]', '', w.lower()) and not is_stop_word(re.sub(r'[^\w]', '', w.lower())) and len(re.sub(r'[^\w]', '', w.lower())) <= 30
            ]
            all_words.extend(extra)
            clean_words.extend(extra)
        def markov_generate(chain, start=None, length=8):
            if not chain:
                return []
            keys = list(chain.keys())
            if not keys:
                return []
            word = start or random.choice(keys)
            phrase = [word]
            for _ in range(length - 1):
                next_words = chain.get(word)
                if not next_words:
                    word = random.choice(keys)
                else:
                    word = random.choice(next_words)
                phrase.append(word)
            return phrase
        rus_words = [w for w in all_words if not re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', w)]
        rus_words = [w for w in rus_words if not is_stop_word(w)]
        if rus_words and markov_chain:
            start_word = random.choice(rus_words)
            markov_phrase = markov_generate(markov_chain, start=start_word, length=random.randint(5, 9))
        else:
            markov_phrase = rus_words[:random.randint(5, 9)]
        mixed = []
        for w in markov_phrase:
            jp = japanese_vocab.get(w)
            if jp and random.random() < 0.5:
                mixed.append(jp)
                if random.random() < 0.3:
                    mixed.append(w)
            else:
                mixed.append(w)
        if random.random() < 0.18 and japanese_vocab:
            mixed.append(random.choice(list(japanese_vocab.values())))
        if rus_words:
            if random.random() < 0.7:
                mixed.insert(0, random.choice(rus_words))
            if random.random() < 0.7:
                mixed.append(random.choice(rus_words))
        base_phrase = " ".join(mixed).strip()
        agent = MAE.select_agent()
        troll_phrase = await agent.speak(base_phrase, detected_lang)

        # --- Определяем язык исходного сообщения пользователя ---
        # detected_lang already set above (unified)

        # --- Если язык русский, ограничиваем до 5% японских слов в troll_phrase для текста ---
        troll_phrase_for_text = troll_phrase
        # Full Japanese mode if chat is in Japanese
        if detected_lang == "ja":
            troll_phrase_for_text = troll_phrase  # allow full JP
        elif detected_lang in ["ru", "en", "fr"]:
            words = troll_phrase.split()
            total_words = len(words)
            # Максимум 5% японских слов, минимум 1 если фраза короткая
            max_jp = max(1, int(total_words * 0.05 + 0.5))
            # Индексы японских слов
            jp_indices = [i for i, w in enumerate(words) if re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', w)]
            if len(jp_indices) > max_jp:
                # Оставим только первые max_jp японских слов, остальные заменим на русские аналоги если есть, или уберём
                jp_keep = set(jp_indices[:max_jp])
                new_words = []
                for i, w in enumerate(words):
                    if re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', w):
                        if i in jp_keep:
                            new_words.append(w)
                        else:
                            rus = jp_rus_map.get(w)
                            if rus:
                                new_words.append(rus)
                            # иначе пропускаем японское слово
                    else:
                        new_words.append(w)
                troll_phrase_for_text = " ".join(new_words)
            else:
                troll_phrase_for_text = troll_phrase
        else:
            # If not Russian/English/French — keep default but reduce JP to 30%
            words = troll_phrase.split()
            new_words = []
            for w in words:
                if re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', w) and random.random() > 0.3:
                    # replace extra JP word if exists russian equivalent
                    rus = jp_rus_map.get(w)
                    if rus:
                        new_words.append(rus)
                else:
                    new_words.append(w)
            troll_phrase_for_text = " ".join(new_words)

        # --- Используем язык чата для TTS, японский только как вставка ---
        # final_audio_text — текст на языке чата
        words = troll_phrase_for_text.split()
        # вставляем 5–10% японских слов случайно
        num_jp_insert = max(1, int(len(words) * 0.07))
        jp_candidates = list(japanese_vocab.values())
        for _ in range(num_jp_insert):
            idx = random.randint(0, len(words)-1)
            words[idx] = random.choice(jp_candidates)
        final_audio_text = " ".join(words)

        # --- 3. Выбираем случайную стратегию отправки ---
        # 0: только текст, 1: только голос, 2: только мем, 3: текст+мем, 4: голос+мем, 5: текст+голос, 6: всё
        mode = random.choices(
            population=[0, 1, 2, 3, 4, 5, 6],
            weights=[0.18, 0.2, 0.13, 0.18, 0.13, 0.09, 0.09],  # суммарно 1.0
            k=1
        )[0]

        sent_something = False
        errors = []

        # --- Текст ---
        async def send_text():
            if not hasattr(update, "message") or not update.message:
                logger.warning("send_text: update.message отсутствует, пропускаем")
                return False
            try:
                # Обрезаем troll_phrase до максимум двух слов
                short_troll_phrase = " ".join(troll_phrase_for_text.split()[:2])
                # Используем безопасную отправку с retry
                success = await safe_reply_text(update.message, short_troll_phrase)
                if not success:
                    logger.error("send_text: все попытки safe_reply_text не удались")
                return success
            except Exception as e:
                logger.error(f"troll_text: send_text unexpected error: {e}")
                return False

        # --- Голос ---
        async def send_voice():
            try:
                # Определяем язык голоса
                try:
                    detected = detect(final_audio_text)
                except Exception:
                    detected = None
                voice_lang = detected if detected in ["ja", "ru", "en", "fr"] else detected_lang

            # Подготовка текста для TTS
                audio_text_words = final_audio_text.split()
                if voice_lang in ["ru", "en", "fr"]:
                    filtered_words = []
                    for w in audio_text_words:
                        if re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', w):
                            rus = jp_rus_map.get(w)
                            if rus:
                                filtered_words.append(rus)
                        else:
                            filtered_words.append(w)
                    voice_text = " ".join(filtered_words).strip()
                else:
                    voice_text = final_audio_text.strip()
                if len(voice_text) < 3:
                    voice_text = "にゃん"

         # Получаем текущее состояние MutRes для pitch/volume
                try:
                    mutres = MutRes()
                    current_state = mutres.state.copy()
                    res_energy = float(np.mean(np.abs(current_state)))
                    pitch_shift = max(-2.0, min(2.0, (res_energy - 0.5) * 4.0))
                    volume_mod = 1.0 + (res_energy - 0.5) * 0.6
                except Exception as e:
                    logger.warning(f"MutRes integration in voice failed: {e}")
                    pitch_shift = 0.0
                    volume_mod = 1.0

                # Генерация аниме-голоса через make_anime_voice
                def tts_generate(text, lang, pitch_shift=0.0, volume_mod=1.0):
                    audio = make_anime_voice(text, voice_lang=lang)
                    # Применяем pitch/volume модуляцию
                    orig_rate = audio.frame_rate
                    new_rate = int(orig_rate * (2.0 ** (pitch_shift / 12.0)))
                    audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_rate})
                    audio = audio.set_frame_rate(22050)
                    audio += (volume_mod - 1.0) * 5.0
                    return audio

                audio_segment = await asyncio.to_thread(tts_generate, voice_text, voice_lang, pitch_shift, volume_mod)

                # Экспорт в ogg для Telegram
                buf_out = io.BytesIO()
                audio_segment.export(buf_out, format="ogg", codec="libopus", bitrate="48k")
                buf_out.seek(0)
                duration = int(len(audio_segment) / 1000)

                await update.message.reply_voice(
                    voice=InputFile(buf_out, f"yuma_voice_{voice_lang}.ogg"),
                    duration=duration
                )
                return True
            except Exception as e:
                logger.error(f"troll_text send_voice error: {e}")
                await update.message.reply_text("… (голос сломался)")
                return False

        # --- Мем ---
        async def send_meme():
            try:
                await generate_meme(update, context)
                return True
            except Exception as e:
                logger.error(f"troll_text: send_meme error: {e}")
                errors.append("meme")
                return False

        # --- Выполняем действия по mode ---
        if mode == 0:
            await send_text()
        elif mode == 1:
            await send_voice()
        elif mode == 2:
            await send_meme()
        elif mode == 3:
            await send_text()
            await send_meme()
        elif mode == 4:
            await send_voice()
            await send_meme()
        elif mode == 5:
            await send_text()
            await send_voice()
        elif mode == 6:
            await send_text()
            await send_voice()
            await send_meme()
        # Apply basic reward: user interaction from troll
        reward = {"user_interaction": True}
        MAE.apply_reward(reward)
    except Exception as e:
        logger.error(f"troll_text: {e}")
        await update.message.reply_text("にゃん… 声が壊ен…")

# —————————–
# Команды
# —————————–
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "にゃっはー！<b>Yuma Nami v3.2</b> 起動！\n"
        "Чат + Reddit → японский голос. Никаких шаблонов.\n\n"
        "<b>Команды:</b>\n"
        "/troll — тролль\n"
        "/status — статус\n"
        "/reset_memory — сброс\n"
        "/set_threshold &lt;число&gt; — порог\n"
        "/fetch_reddit — Reddit сейчас",
        parse_mode='HTML'
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    total_energy = sum(word_weights.values())
    meta = yuma_identity["meta_analysis"]
    msg = (
        f"<b>Yuma Status</b>\n\n"
        f"Сообщений: <code>{len(recent_messages)}</code>\n"
        f"Слов: <code>{len(word_weights)}</code>\n"
        f"Энергия: <code>{total_energy}/{RESO_THRESHOLD}</code>\n"
        f"Японский словарь: <code>{len(japanese_vocab)}</code>\n"
        f"Reddit: <code>{len(reddit_meme_texts)}</code> текстов\n"
        f"Эмоция: <code>{meta['dominant_emotions'].get('dominant', '—')}</code>"
    )
    await safe_reply_text(update.message, msg, parse_mode='HTML')

# --- Evolution command ---
async def evolution(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = "<b>Evolution Status</b>\n\n"
        
        # Сортировка агентов по энергии, по убыванию
        sorted_agents = sorted(MAE.agents, key=lambda x: getattr(x, "energy", 0), reverse=True)

        for a in sorted_agents:
            emoji = getattr(a, "style_emoji", "—")
            ratio = getattr(a, "jp_ratio", None)
            if isinstance(ratio, float):
                ratio_display = f"{ratio:.2f}"
            else:
                ratio_display = "—"
            
            # Прогресс-бар энергии (0-100)
            energy = getattr(a, "energy", 0)
            max_energy = getattr(a, "max_energy", 100)
            filled = int((energy / max_energy) * 10)  # 10 символов
            bar = "█" * filled + "░" * (10 - filled)

            # Статус или настроение агента
            status = getattr(a, "status", "Idle")
            
            msg += (
                f"<code>{a.name}</code> | "
                f"{bar} <b>{energy}/{max_energy}</b> | "
                f"Emoji: {emoji} | "
                f"JP%: {ratio_display} | "
                f"Status: {status}\n"
            )
        
        await update.message.reply_text(msg, parse_mode='HTML')
    except Exception as e:
        logger.error(f"evolution cmd: {e}")
        await update.message.reply_text("Evolution error…")

async def reset_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    init_data()
    await update.message.reply_text("Память стёрта. Хаос перезагружен...")

async def set_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("<b>/set_threshold 30</b>", parse_mode='HTML')
        return
    global RESO_THRESHOLD
    RESO_THRESHOLD = int(context.args[0])
    save_data()
    await update.message.reply_text(f"Порог: <b>{RESO_THRESHOLD}</b>", parse_mode='HTML')

async def fetch_reddit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Reddit парсинг... [loading]")
    memes = await fetch_reddit_json()
    if not memes:
        memes = await fetch_reddit_fallback()
    integrate_reddit_memes(memes)
    await update.message.reply_text(f"Готово! +{len(memes)} мемов. Хаос усилен! ✨")

# —————————–
# ==============================================================================
# Resonance Synchronization Protocol (Experimental Multi-Node Network Section)
# ==============================================================================
# This section implements the basis for a resonance synchronization protocol
# across multiple Yuma nodes using WebSockets. Each node broadcasts and receives
# resonance packets containing its resonance state, and merges incoming packets
# into its local resonance field.
#
# Packet fields: node_id, timestamp, energy_vector, entropy_level, dominant_emotion
#
# Requirements: websockets (pip install websockets), asyncio

# mutres_core_async.py
import os
import uuid
import time
import json
import logging
import asyncio
import numpy as np
import websockets

class MutRes:
    """
    Асинхронное резонансное ядро, неблокирующее и лёгкое по ресурсам.
    - Отложенный старт: не создаёт таск в __init__, вместо этого предоставляет async start().
    - Использует asyncio primitives для безопасности и мягкой остановки.
    - Минимизирует работу в синхронных вызовах: get_state() возвращает копию массива без блокировок.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, state_size=10, decay=0.95, update_interval=0.12):
        # Инициализируем только один раз
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self.state_size = int(state_size)
        self.decay = float(decay)
        self.update_interval = float(update_interval)

        # Используем fast PRNG
        self._rng = np.random.default_rng()
        self._state = np.zeros(self.state_size, dtype=float)

        # Защита доступа и управление таском
        self._lock = asyncio.Lock()
        self._task = None
        self._stop_event = asyncio.Event()

        # callbacks may be sync or async - keep small
        self.callbacks = []

        # lightweight logger
        self._log = logging.getLogger("MutRes")
        self._log.setLevel(logging.WARNING)

    async def start(self):
        """Start background update loop. Safe to call multiple times."""
        if self._task is not None and not self._task.done():
            return
        self._stop_event.clear()
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self._loop(), name="MutRes._loop")

    async def stop(self):
        """Signal background loop to stop and wait for it to finish."""
        self._stop_event.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except asyncio.TimeoutError:
                try:
                    self._task.cancel()
                except Exception:
                    pass
        self._task = None

    async def _loop(self):
        """Background coroutine that updates state asynchronously and calls callbacks.
        It keeps work minimal and yields to the event loop frequently.
        """
        try:
            while not self._stop_event.is_set():
                await self._update_state_once()
                # sleep is the main yield point; keep interval modest
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self.update_interval)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            # Never raise from background loop — just log
            self._log.warning(f"MutRes loop exception: {e}")

    async def _update_state_once(self):
        """One non-blocking update of internal state. Keeps lock time very short."""
        # Create tiny noise vector using numpy rng (fast)
        noise = (self._rng.random(self.state_size) - 0.5) * 0.02
        # Do arithmetic outside lock, then swap in under lock
        new_state = self._state * self.decay + noise
        # Short critical section
        async with self._lock:
            self._state = new_state
            state_copy = self._state.copy()
        # Call callbacks but do not await them here — schedule them to run later
        for cb in list(self.callbacks):
            try:
                res = cb(state_copy)
                if asyncio.iscoroutine(res):
                    # schedule coroutine but don't await
                    asyncio.create_task(res)
            except Exception as e:
                self._log.warning(f"MutRes callback error: {e}")

    def attach(self, func):
        """Attach a callback (sync or async). Callbacks will be scheduled after updates."""
        if callable(func):
            self.callbacks.append(func)

    def detach(self, func):
        try:
            self.callbacks.remove(func)
        except ValueError:
            pass

    def get_state(self):
        """Return a thread-safe copy of current state (sync). Very cheap."""
        # No await required; do a fast local copy under lock if loop is running
        if self._lock.locked():
            # If lock is locked, try a non-blocking approach
            return self._state.copy()
        # Safe copy
        return self._state.copy()

    @property
    def state(self):
        return self.get_state()

    # Backwards-compatible stop method name
    def stop_sync(self):
        # schedule stop in background
        try:
            asyncio.create_task(self.stop())
        except Exception:
            pass


# --- Автономная инициализация ---
mutres = MutRes()
# schedule the MutRes background loop safely in the running event loop
try:
    asyncio.create_task(mutres.start())
except Exception:
    # If we're not in an event loop at import time, the main() will start it explicitly
    pass

# Unique node ID (persist or random per session)
NODE_ID = os.environ.get("YUMA_NODE_ID") or str(uuid.uuid4())

# Local resonance field (can be more sophisticated)
local_resonance_field = {
    "energy_vector": [0.0, 0.0, 0.0, 0.0],  # e.g., [joy, tension, flow, surprise]
    "entropy_level": 0.0,
    "dominant_emotion": "flow"
}

def build_resonance_packet():
    """
    Build a resonance packet for broadcasting.
    Returns a dict ready for JSON serialization.
    """
    packet = {
        "node_id": NODE_ID,
        "timestamp": time.time(),
        "energy_vector": list(local_resonance_field.get("energy_vector", [0.0, 0.0, 0.0, 0.0])),
        "entropy_level": float(local_resonance_field.get("entropy_level", 0.0)),
        "dominant_emotion": str(local_resonance_field.get("dominant_emotion", "flow"))
    }
    return packet

def update_resonance_field(packet):
    """
    Merge incoming resonance packet into local resonance.
    This is a naive merge: weighted average for energy_vector/entropy, update dominant_emotion by majority.
    """
    if not isinstance(packet, dict) or packet.get("node_id") == NODE_ID:
        return
    try:
        # Weighted average with incoming
        lv = local_resonance_field.get("energy_vector", [0.0, 0.0, 0.0, 0.0])
        pv = packet.get("energy_vector", [0.0, 0.0, 0.0, 0.0])
        local_resonance_field["energy_vector"] = [
            (a + b) / 2.0 for a, b in zip(lv, pv)
        ]
        le = float(local_resonance_field.get("entropy_level", 0.0))
        pe = float(packet.get("entropy_level", 0.0))
        local_resonance_field["entropy_level"] = (le + pe) / 2.0
        # Dominant emotion: majority voting (for demo, just use incoming)
        local_resonance_field["dominant_emotion"] = packet.get("dominant_emotion", local_resonance_field.get("dominant_emotion"))
    except Exception as e:
        logger.warning(f"Resonance sync merge error: {e}")

async def resonance_sync_loop(
    uri="ws://localhost:8765",
    interval=5.0
):
    """
    Periodically broadcasts and receives resonance packets over WebSockets.
    This is a basic loop: connects to a WebSocket server, sends local state, receives others'.
    """
    while True:
        try:
            async with websockets.connect(uri) as ws:
                logger.info(f"[ResonanceSync] Connected to {uri}")
                while True:
                    # Send local resonance packet
                    packet = build_resonance_packet()
                    await ws.send(json.dumps(packet))
                    # Try to receive one or more packets
                    try:
                        resp = await asyncio.wait_for(ws.recv(), timeout=interval)
                        if resp:
                            try:
                                incoming = json.loads(resp)
                                update_resonance_field(incoming)
                                logger.debug(f"[ResonanceSync] Merged packet from {incoming.get('node_id')}")
                            except Exception as e:
                                logger.warning(f"Resonance sync JSON error: {e}")
                    except asyncio.TimeoutError:
                        pass  # No packet received this interval
                    await asyncio.sleep(interval)
        except Exception as e:
            logger.warning(f"[ResonanceSync] Connection error: {e}. Retrying in 10s.")
            await asyncio.sleep(10)

# To start the resonance sync loop, call:
# asyncio.create_task(resonance_sync_loop("ws://your_server:8765"))
# This is only a demo; in production, use a real WebSocket server and robust error handling.
# Запуск
# —————————–
async def main():
    app = Application.builder().token("YourTokenHere").build()
    await app.initialize()
    WEBAPP_URL = "https://0penagi.github.io/YunaNami/"
# в handler start:
    from telegram import WebAppInfo, InlineKeyboardMarkup, InlineKeyboardButton
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("troll", troll_text))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("evolution", evolution))
    app.add_handler(CommandHandler("reset_memory", reset_memory))
    app.add_handler(CommandHandler("set_threshold", set_threshold))
    app.add_handler(CommandHandler("fetch_reddit", fetch_reddit))
    app.add_handler(MessageHandler(filters.PHOTO | (filters.TEXT & ~filters.COMMAND), collect_words))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    # --- Автономная инициализация ---
    mutres = MutRes()
    asyncio.create_task(mutres.start())
    asyncio.create_task(auto_reddit_fetch())
    asyncio.create_task(auto_rss_fetch())
    logger.info("Yuma Nami v3.2 — ПОЛНЫЙ АСИНХРОННЫЙ БОТ ЗАПУЩЕН")
    try:
        await app.run_polling()
    except RuntimeError as e:
        if "Cannot close a running event loop" in str(e):
            pass
        else:
            raise

if __name__ == '__main__':
    import nest_asyncio
    nest_asyncio.apply()
    load_data()
    load_ltm_pt()  # Загружаем память из .pt при старте, если есть
    import asyncio
    asyncio.run(main())
