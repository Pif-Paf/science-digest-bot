#!/usr/bin/env python3
"""
Science Digest Bot (OpenRouter Edition) — ежедневный дайджест по ИБ, ML и AI.

Полностью бесплатный стек:
- Сбор: arXiv, Hacker News (открытые API, без ключей)
- Анализ: OpenRouter free-модели (DeepSeek V3, Llama 3.3 70B и др.) с fallback-цепочкой
- Озвучка: edge-tts (Microsoft, бесплатно)
- Доставка: Telegram (текст + голосовое сообщение)
- Запуск: GitHub Actions по расписанию

Дайджест формируется НА РУССКОМ языке.
"""

import os
import sys
import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import requests
import feedparser

# ============================================================================
# КОНФИГУРАЦИЯ (значения берутся из переменных окружения / GitHub Secrets)
# ============================================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# OpenRouter endpoint (OpenAI-совместимый)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Fallback-цепочка бесплатных моделей (в порядке приоритета).
# Бот пробует их по очереди: если модель недоступна/перегружена — берёт следующую.
#
# ВАЖНО: список free-моделей на OpenRouter меняется со временем. Если какая-то
# модель пропадёт из бесплатного тира, проверьте актуальные слаги здесь:
# https://openrouter.ai/models (включите фильтр :free) и обновите список ниже.
#
# Примечание про :free — у большинства моделей бесплатный вариант помечается
# суффиксом ":free". Исключение — "нативно бесплатные" модели самого OpenRouter
# (например, owl-alpha и openrouter/free), у них суффикса нет — так и должно быть.
#
# Примечание про приватность — preview/alpha-модели (Owl Alpha, Laguna и др.)
# обычно логируют промпты и ответы для улучшения модели. Для научного дайджеста
# без приватных данных это приемлемо, но имейте в виду.
MODEL_FALLBACK_CHAIN = [
    "nvidia/nemotron-3-ultra-550b-a55b:free", # NVIDIA Nemotron 3 Ultra 550B, 1M контекст
    "poolside/laguna-m.1:free",               # Poolside Laguna M.1
    "openai/gpt-oss-120b:free",               # OpenAI GPT-OSS 120B
    "openrouter/owl-alpha",                   # Owl Alpha 
    "openrouter/free",                        # страховка: авто-выбор любой доступной free-модели
]

# Озвучка
ENABLE_AUDIO = True                       # True = генерировать и отправлять аудио
TTS_VOICE = "ru-RU-SvetlanaNeural"        # женский голос; мужской: ru-RU-DmitriyNeural
TTS_MAX_CHARS = 3000                      # ограничение длины текста для озвучки

# ============================================================================
# БЛОК 1: СБОР СТАТЕЙ ИЗ ARXIV
# ============================================================================

def fetch_arxiv_papers(categories: List[str], max_results: int = 5) -> List[Dict]:
    """
    Собирает свежие статьи с arXiv по интересующим категориям.

    Категории:
    - cs.CR  — Cryptography and Security (информационная безопасность)
    - cs.LG  — Machine Learning
    - cs.AI  — Artificial Intelligence
    """
    papers = []
    base_url = "https://export.arxiv.org/api/query"
    # Граница свежести: статьи за последние ~2 суток (с запасом, т.к. arXiv
    # обновляется пачками и узкое окно может давать пустой результат)
    cutoff = datetime.utcnow() - timedelta(days=2)

    for category in categories:
        # ВАЖНО: фильтр submittedDate:[...] в search_query у arXiv нестабилен
        # и часто отдаёт HTTP 500. Поэтому НЕ фильтруем по дате в запросе, а
        # берём самые свежие статьи сортировкой и отсекаем старые уже у себя.
        params = {
            "search_query": f"cat:{category}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max_results * 2,  # берём с запасом под фильтрацию по дате
            "start": 0,
        }

        try:
            headers = {"User-Agent": "Science-Digest-Bot/2.0 (personal research use)"}
            response = requests.get(base_url, params=params, headers=headers, timeout=20)
            response.raise_for_status()
            feed = feedparser.parse(response.content)

            added = 0
            for entry in feed.entries:
                if added >= max_results:
                    break
                # Отсекаем старые статьи (если дата распознаётся)
                try:
                    pub = datetime(*entry.published_parsed[:6])
                    if pub < cutoff:
                        continue
                except Exception:
                    pass  # если дату не распознали — оставляем статью

                papers.append({
                    "source": "arXiv",
                    "title": entry.title.replace("\n", " ").strip(),
                    "authors": ", ".join(a.name for a in entry.authors[:3]) if hasattr(entry, "authors") else "",
                    "summary": entry.summary.replace("\n", " ").strip()[:400],
                    "url": entry.id,
                    "published": getattr(entry, "published", ""),
                    "category": category,
                })
                added += 1
            # Лёгкая пауза, чтобы не упереться в rate limit arXiv
            time.sleep(3)  # arXiv просит не чаще 1 запроса в 3 сек
        except Exception as e:
            print(f"[arXiv:{category}] ошибка: {e}", file=sys.stderr)

    return papers


# ============================================================================
# БЛОК 2: СБОР ИСТОРИЙ И КОММЕНТАРИЕВ ИЗ HACKER NEWS
# ============================================================================

def fetch_hackernews_comments(story_id: int, max_comments: int = 4) -> List[str]:
    """
    Собирает несколько верхних комментариев к истории Hacker News.
    HN отдаёт ID комментариев в поле 'kids'; берём первые (они отсортированы
    по рейтингу) и тянем их текст. Возвращает список текстов комментариев.
    """
    comments = []
    try:
        item_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        resp = requests.get(item_url, timeout=10)
        resp.raise_for_status()
        story = resp.json()

        kid_ids = story.get("kids", [])[:max_comments + 3]
        for kid_id in kid_ids:
            if len(comments) >= max_comments:
                break
            try:
                c_url = f"https://hacker-news.firebaseio.com/v0/item/{kid_id}.json"
                c_resp = requests.get(c_url, timeout=10)
                c_resp.raise_for_status()
                comment = c_resp.json()
                if comment.get("deleted") or comment.get("dead"):
                    continue
                text = comment.get("text", "")
                if text:
                    # HN хранит комментарии в HTML — грубо чистим теги
                    import re
                    clean = re.sub(r"<[^>]+>", " ", text)
                    clean = clean.replace("&#x27;", "'").replace("&quot;", '"')
                    clean = clean.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
                    comments.append(clean.strip()[:300])
            except Exception:
                continue
    except Exception as e:
        print(f"[HN comments] ошибка для {story_id}: {e}", file=sys.stderr)

    return comments


def fetch_hackernews_stories(max_stories: int = 5, with_comments: bool = True) -> List[Dict]:
    """
    Собирает топовые истории Hacker News, фильтруя по ключевым словам тематики.
    Официальный API, полностью бесплатный. Опционально подтягивает комментарии
    к самым обсуждаемым историям (для анализа реакции сообщества).
    """
    stories = []
    keywords = [
        "ml", "ai", "machine learning", "deep learning", "neural",
        "security", "exploit", "vulnerability", "crypto", "llm",
        "model", "attack", "privacy", "cyber",
    ]

    try:
        top_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
        response = requests.get(top_url, timeout=15)
        response.raise_for_status()
        top_ids = response.json()[:40]  # берём с запасом для фильтрации

        for story_id in top_ids:
            if len(stories) >= max_stories:
                break
            try:
                item_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                item_resp = requests.get(item_url, timeout=10)
                item_resp.raise_for_status()
                story = item_resp.json()

                title_lower = story.get("title", "").lower()
                if any(kw in title_lower for kw in keywords):
                    stories.append({
                        "source": "Hacker News",
                        "title": story.get("title", ""),
                        "author": story.get("by", "unknown"),
                        "score": story.get("score", 0),
                        "url": story.get("url", f"https://news.ycombinator.com/item?id={story_id}"),
                        "num_comments": story.get("descendants", 0),
                        "hn_url": f"https://news.ycombinator.com/item?id={story_id}",
                        "_story_id": story_id,
                    })
            except Exception:
                continue

        # Подтягиваем комментарии к самым обсуждаемым историям
        if with_comments:
            top_discussed = sorted(stories, key=lambda s: s.get("num_comments", 0), reverse=True)
            for story in top_discussed[:3]:  # только к 3 самым обсуждаемым
                if story.get("num_comments", 0) > 5:
                    story["top_comments"] = fetch_hackernews_comments(
                        story["_story_id"], max_comments=4
                    )

        # Убираем служебное поле перед передачей в модель
        for story in stories:
            story.pop("_story_id", None)

    except Exception as e:
        print(f"[Hacker News] ошибка: {e}", file=sys.stderr)

    return stories


# ============================================================================
# БЛОК 3: АНАЛИЗ ЧЕРЕЗ OPENROUTER (с fallback-цепочкой моделей)
# ============================================================================

def call_openrouter(prompt: str, max_tokens: int = 2000) -> Optional[str]:
    """
    Отправляет запрос в OpenRouter, перебирая модели из MODEL_FALLBACK_CHAIN.
    Возвращает текст ответа первой успешной модели или None, если все упали.

    Реализован простой backoff на случай rate limit (429).
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # OpenRouter рекомендует указывать эти заголовки (необязательно):
        "HTTP-Referer": "https://github.com/science-digest-bot",
        "X-Title": "Science Digest Bot",
    }

    for model in MODEL_FALLBACK_CHAIN:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.4,
        }

        # До 3 попыток на модель (на случай временного 429)
        for attempt in range(3):
            try:
                print(f"   → пробую модель: {model} (попытка {attempt + 1})")
                response = requests.post(OPENROUTER_URL, headers=headers,
                                         json=payload, timeout=120)

                if response.status_code == 429:
                    wait = 5 * (attempt + 1)
                    print(f"     429 rate limit, жду {wait}с...")
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()

                # Защита от пустого/нестандартного ответа
                if "choices" in data and data["choices"]:
                    content = data["choices"][0]["message"]["content"]
                    if content and content.strip():
                        print(f"     ✓ успешно через {model}")
                        return content.strip()

                print(f"     пустой ответ от {model}, перехожу к следующей")
                break

            except requests.exceptions.HTTPError as e:
                print(f"     HTTP ошибка {model}: {e}")
                break  # пробуем следующую модель
            except Exception as e:
                print(f"     ошибка {model}: {e}")
                time.sleep(3)

    print("   ✗ Все модели в цепочке недоступны", file=sys.stderr)
    return None


def analyze_materials(all_materials: List[Dict]) -> str:
    """
    Формирует промт для модели и получает структурированный дайджест НА РУССКОМ.
    """
    materials_text = json.dumps(all_materials, ensure_ascii=False, indent=2)

    prompt = f"""Ты — персональный научный ассистент русскоязычного исследователя в области \
информационной безопасности, машинного обучения и искусственного интеллекта.

Перед тобой материалы за последние сутки: научные статьи (arXiv) и обсуждения в \
сообществе (Hacker News), включая комментарии пользователей.

ИСХОДНЫЕ МАТЕРИАЛЫ (в формате JSON):
{materials_text}

ТВОЯ ЗАДАЧА — составить дайджест НА РУССКОМ ЯЗЫКЕ:

1. Отбери 5-7 НАИБОЛЕЕ значимых материалов по критериям:
   - новизна и актуальность;
   - практическая важность для ИБ / ML / AI;
   - качество источника (рецензируемая статья > препринт > обсуждение).

2. Для каждого отобранного материала укажи:
   - заголовок (если оригинал на английском — дай русский перевод, а в скобках оригинал);
   - источник и прямую ссылку (URL);
   - суть в 2-3 предложениях своими словами;
   - почему это важно;
   - если есть комментарии/обсуждение — краткий анализ реакции сообщества \
(основные позиции, согласие/несогласие, критика, ценные дополнения).

3. В конце добавь раздел «ГЛАВНЫЕ ВЫВОДЫ ДНЯ» — 3 ключевых тезиса.

ТРЕБОВАНИЯ К ФОРМАТУ (используй именно такую разметку):

🔝 ТОП НАХОДОК ДНЯ

1. [Заголовок]
Источник: [...] | Ссылка: [URL]
Суть: [...]
Почему важно: [...]
Реакция сообщества: [... или «обсуждений пока нет»]

(и так далее по каждому материалу)

📌 ГЛАВНЫЕ ВЫВОДЫ ДНЯ
1. ...
2. ...
3. ...

Будь точен и критичен. Не выдумывай факты, которых нет в материалах. \
Если материалов мало или они слабые — честно отметь это. \
Пиши живым, профессиональным русским языком."""

    print("\n🤖 Отправляю материалы в OpenRouter на анализ...")
    result = call_openrouter(prompt, max_tokens=2500)

    if result is None:
        return ("❌ Не удалось получить анализ: все бесплатные модели OpenRouter "
                "сейчас недоступны или перегружены. Попробуйте позже или проверьте "
                "актуальность списка моделей на https://openrouter.ai/models")

    return result


# ============================================================================
# БЛОК 4: ОЗВУЧКА (EDGE-TTS + ЗАПАСНОЙ GTTS) (бесплатно)
# ============================================================================

def _prepare_tts_text(text: str) -> str:
    """Чистит текст от разметки и эмодзи, ограничивает длину — общий для обоих движков."""
    clean = (text
             .replace("🔝", "").replace("📌", "").replace("**", "")
             .replace("#", "").replace("*", "").replace("━", ""))
    clean = clean[:TTS_MAX_CHARS]
    intro = f"Научный дайджест за {datetime.now().strftime('%d.%m.%Y')}. "
    return intro + clean


async def _synthesize_edge(speech_text: str, output_file: str) -> bool:
    """Основной движок: Microsoft Edge TTS. Возвращает True при успехе."""
    try:
        import edge_tts
    except ImportError:
        print("   ⚠️  edge-tts не установлен, пробую запасной движок", file=sys.stderr)
        return False

    try:
        communicate = edge_tts.Communicate(speech_text, TTS_VOICE)
        await communicate.save(output_file)
        # Проверяем, что файл не пустой (edge-tts иногда «успешно» создаёт 0 байт)
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"   ✓ Аудио (edge-tts) сохранено: {output_file}")
            return True
        print("   ⚠️  edge-tts вернул пустой файл", file=sys.stderr)
        return False
    except Exception as e:
        print(f"   ⚠️  edge-tts недоступен: {e}", file=sys.stderr)
        return False


def _synthesize_gtts(speech_text: str, output_file: str) -> bool:
    """Запасной движок: Google TTS (gTTS). Другой сервис — не падает вместе с Edge."""
    try:
        from gtts import gTTS
    except ImportError:
        print("   ⚠️  gTTS не установлен, аудио пропущено", file=sys.stderr)
        return False

    try:
        tts = gTTS(text=speech_text, lang="ru")
        tts.save(output_file)
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"   ✓ Аудио (gTTS, запасной движок) сохранено: {output_file}")
            return True
        return False
    except Exception as e:
        print(f"   ⚠️  gTTS тоже недоступен: {e}", file=sys.stderr)
        return False


async def synthesize_audio(text: str, output_file: str = "digest.mp3") -> Optional[str]:
    """
    Озвучивает дайджест. Сначала пробует Microsoft Edge TTS, при сбое (например,
    частая ошибка 403) автоматически переключается на запасной движок gTTS (Google).
    Возвращает путь к файлу или None, если оба движка не сработали.
    """
    speech_text = _prepare_tts_text(text)

    # 1) Основной движок — Edge TTS (качественный русский голос)
    if await _synthesize_edge(speech_text, output_file):
        return output_file

    # 2) Запасной движок — gTTS (другой сервис, выручает при 403 от Edge)
    print("   → переключаюсь на запасной движок gTTS...")
    if _synthesize_gtts(speech_text, output_file):
        return output_file

    print("   ⚠️  Оба TTS-движка недоступны, дайджест отправлен только текстом",
          file=sys.stderr)
    return None


# ============================================================================
# БЛОК 5: ОТПРАВКА В TELEGRAM (текст + аудио)
# ============================================================================

def send_telegram_text(text: str) -> bool:
    """Отправляет текстовое сообщение (с авторазбиением на части по 4096 символов)."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID не заданы.", file=sys.stderr)
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    max_len = 4096
    chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)]

    try:
        for chunk in chunks:
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": chunk,
                "disable_web_page_preview": True,
            }
            r = requests.post(url, json=payload, timeout=20)
            r.raise_for_status()
            time.sleep(0.5)
        print(f"✅ Текст отправлен в Telegram ({len(chunks)} сообщ.)")
        return True
    except Exception as e:
        print(f"❌ Ошибка отправки текста в Telegram: {e}", file=sys.stderr)
        return False


def send_telegram_audio(file_path: str) -> bool:
    """Отправляет MP3-файл как аудио в Telegram."""
    if not file_path or not os.path.exists(file_path):
        return False
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendAudio"
    try:
        with open(file_path, "rb") as audio:
            files = {"audio": audio}
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "title": f"Научный дайджест {datetime.now().strftime('%d.%m.%Y')}",
            }
            r = requests.post(url, data=payload, files=files, timeout=60)
            r.raise_for_status()
        print("✅ Аудио отправлено в Telegram")
        return True
    except Exception as e:
        print(f"❌ Ошибка отправки аудио: {e}", file=sys.stderr)
        return False


# ============================================================================
# БЛОК 6: ОСНОВНОЙ WORKFLOW
# ============================================================================

def main():
    print(f"🚀 Science Digest Bot (OpenRouter) — {datetime.now().isoformat()}")
    print("-" * 60)

    # --- Шаг 1: сбор материалов ---
    print("📚 arXiv...")
    arxiv_papers = fetch_arxiv_papers(["cs.CR", "cs.LG", "cs.AI"], max_results=6)
    print(f"   ✓ {len(arxiv_papers)} статей")

    print("📰 Hacker News (истории + комментарии)...")
    hn_stories = fetch_hackernews_stories(max_stories=8, with_comments=True)
    print(f"   ✓ {len(hn_stories)} историй")

    all_materials = arxiv_papers + hn_stories
    print(f"\n📊 Всего материалов: {len(all_materials)}")

    if not all_materials:
        print("⚠️  Нет материалов, выходим.")
        send_telegram_text("⚠️ Сегодня не удалось собрать материалы для дайджеста.")
        return

    # --- Шаг 2: анализ через OpenRouter ---
    digest = analyze_materials(all_materials)

    print("\n" + "=" * 60)
    print(digest)
    print("=" * 60)

    # --- Шаг 3: формируем итоговое сообщение ---
    full_message = (
        f"🔬 НАУЧНЫЙ ДАЙДЖЕСТ\n"
        f"{datetime.now().strftime('%d.%m.%Y %H:%M')}\n"
        f"━━━━━━━━━━━━━━━━━\n\n"
        f"{digest}\n\n"
        f"━━━━━━━━━━━━━━━━━\n"
        f"🤖 Science Digest Bot (OpenRouter)"
    )

    # --- Шаг 4: отправляем текст ---
    print("\n📤 Отправляю текст в Telegram...")
    send_telegram_text(full_message)

    # --- Шаг 5: озвучка и отправка аудио ---
    if ENABLE_AUDIO:
        print("\n🎙️  Генерирую аудио...")
        audio_file = asyncio.run(synthesize_audio(digest))
        if audio_file:
            send_telegram_audio(audio_file)

    print("\n✅ Готово!")


if __name__ == "__main__":
    main()
