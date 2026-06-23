#!/usr/bin/env python3
"""
Science Digest Bot v2.0 (OpenRouter + Podcast Edition)
ежедневный дайджест по ИБ, ML и AI + обсуждение в подкаст-стиле.

Источники:
- arXiv (научные статьи)
- Hacker News (обсуждения в сообществе)
- Hugging Face (модели и датасеты)

Анализ: OpenRouter free-модели с fallback-цепочкой
Озвучка: Podcast-style discussion (не просто чтение, а обсуждение)
Доставка: Telegram
Запуск: GitHub Actions

Дайджест формируется НА РУССКОМ языке.
"""

import os
import sys
import json
import time
import asyncio
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import requests
import feedparser

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL_FALLBACK_CHAIN = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "poolside/laguna-m.1:free",
    "openai/gpt-oss-120b:free",
    "openrouter/owl-alpha",
    "openrouter/free",
]

# Озвучка
ENABLE_AUDIO = True
TTS_VOICE = "ru-RU-SvetlanaNeural"
TTS_MAX_CHARS = 5000

# ============================================================================
# БЛОК 1: СБОР СТАТЕЙ ИЗ ARXIV
# ============================================================================

def fetch_arxiv_papers(categories: List[str], max_results: int = 5) -> List[Dict]:
    """Собирает свежие статьи с arXiv."""
    papers = []
    base_url = "https://export.arxiv.org/api/query"
    cutoff = datetime.utcnow() - timedelta(days=2)

    for category in categories:
        params = {
            "search_query": f"cat:{category}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max_results * 2,
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
                try:
                    pub = datetime(*entry.published_parsed[:6])
                    if pub < cutoff:
                        continue
                except Exception:
                    pass

                papers.append({
                    "source": "arXiv",
                    "title": entry.title.replace("\n", " ").strip(),
                    "authors": ", ".join(a.name for a in entry.authors[:3]) if hasattr(entry, "authors") else "",
                    "summary": entry.summary.replace("\n", " ").strip()[:500],
                    "url": entry.id,
                    "published": getattr(entry, "published", ""),
                    "category": category,
                })
                added += 1
            time.sleep(3)
        except Exception as e:
            print(f"[arXiv:{category}] ошибка: {e}", file=sys.stderr)

    return papers


# ============================================================================
# БЛОК 2: HACKER NEWS (истории + комментарии)
# ============================================================================

def fetch_hackernews_comments(story_id: int, max_comments: int = 4) -> List[str]:
    """Собирает топовые комментарии к истории."""
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
    """Собирает топовые истории Hacker News."""
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
        top_ids = response.json()[:40]

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

        if with_comments:
            top_discussed = sorted(stories, key=lambda s: s.get("num_comments", 0), reverse=True)
            for story in top_discussed[:3]:
                if story.get("num_comments", 0) > 5:
                    story["top_comments"] = fetch_hackernews_comments(
                        story["_story_id"], max_comments=4
                    )

        for story in stories:
            story.pop("_story_id", None)

    except Exception as e:
        print(f"[Hacker News] ошибка: {e}", file=sys.stderr)

    return stories


# ============================================================================
# БЛОК 3: HUGGING FACE (модели и датасеты)
# ============================================================================

def fetch_huggingface_trending(max_items: int = 5) -> List[Dict]:
    """
    Собирает трендовые модели и датасеты с Hugging Face.
    Использует неофициальный парсинг (т.к. у HF нет официального бесплатного API для трендов).
    """
    items = []
    
    try:
        # Пробуем получить информацию через неофициальный endpoint
        # HF имеет REST API с информацией о моделях и датасетах
        
        # Модели
        models_url = "https://huggingface.co/api/models?sort=downloads&direction=-1&limit=8"
        resp = requests.get(models_url, timeout=15, headers={"User-Agent": "Science-Digest-Bot/2.0"})
        
        if resp.status_code == 200:
            models_data = resp.json()
            if isinstance(models_data, list):
                for model_info in models_data[:max_items]:
                    try:
                        items.append({
                            "source": "Hugging Face",
                            "type": "model",
                            "title": model_info.get("modelId", ""),
                            "url": f"https://huggingface.co/{model_info.get('modelId', '')}",
                            "summary": model_info.get("description", "")[:300] or "AI/ML модель из Hugging Face",
                            "downloads": model_info.get("downloads", 0),
                            "likes": model_info.get("likes", 0),
                            "tags": model_info.get("tags", []),
                        })
                    except Exception:
                        continue
        
        # Датасеты
        datasets_url = "https://huggingface.co/api/datasets?sort=downloads&direction=-1&limit=8"
        resp_ds = requests.get(datasets_url, timeout=15, headers={"User-Agent": "Science-Digest-Bot/2.0"})
        
        if resp_ds.status_code == 200:
            datasets_data = resp_ds.json()
            if isinstance(datasets_data, list):
                for dataset_info in datasets_data[:max_items]:
                    try:
                        items.append({
                            "source": "Hugging Face",
                            "type": "dataset",
                            "title": dataset_info.get("id", ""),
                            "url": f"https://huggingface.co/datasets/{dataset_info.get('id', '')}",
                            "summary": dataset_info.get("description", "")[:300] or "Датасет из Hugging Face",
                            "downloads": dataset_info.get("downloads", 0),
                            "likes": dataset_info.get("likes", 0),
                            "tags": dataset_info.get("tags", []),
                        })
                    except Exception:
                        continue
    
    except Exception as e:
        print(f"[Hugging Face] ошибка: {e}", file=sys.stderr)

    return items[:max_items]


# ============================================================================
# БЛОК 4: OPENROUTER АНАЛИЗ (с fallback-цепочкой)
# ============================================================================

def call_openrouter(prompt: str, max_tokens: int = 2000) -> Optional[str]:
    """Отправляет запрос в OpenRouter с fallback-цепочкой."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
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

                if "choices" in data and data["choices"]:
                    content = data["choices"][0]["message"]["content"]
                    if content and content.strip():
                        print(f"     ✓ успешно через {model}")
                        return content.strip()

                print(f"     пустой ответ от {model}, перехожу к следующей")
                break

            except requests.exceptions.HTTPError as e:
                print(f"     HTTP ошибка {model}: {e}")
                break
            except Exception as e:
                print(f"     ошибка {model}: {e}")
                time.sleep(3)

    print("   ✗ Все модели в цепочке недоступны", file=sys.stderr)
    return None


def analyze_materials(all_materials: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Анализирует материалы и возвращает:
    1. Структурированный дайджест НА РУССКОМ
    2. Список статей с рейтингом и тегами (для метаданных)
    """
    materials_text = json.dumps(all_materials, ensure_ascii=False, indent=2)

    prompt = f"""Ты — персональный научный ассистент русскоязычного исследователя в области \
информационной безопасности, машинного обучения и искусственного интеллекта.

Перед тобой материалы за последние сутки: научные статьи (arXiv), обсуждения в сообществе \
(Hacker News) и трендовые модели/датасеты (Hugging Face).

ИСХОДНЫЕ МАТЕРИАЛЫ (в формате JSON):
{materials_text}

ТВОЯ ЗАДАЧА — составить дайджест НА РУССКОМ ЯЗЫКЕ:

1. Отбери 5-8 НАИБОЛЕЕ значимых материалов по критериям:
   - новизна и актуальность;
   - практическая важность для ИБ / ML / AI;
   - качество источника;
   - влияние на сообщество (для HN — количество комментариев; для HF — лайки/скачивания).

2. Для каждого отобранного материала укажи:
   - заголовок (если оригинал на английском — дай русский перевод + оригинал в скобках);
   - источник и прямую ссылку (URL);
   - суть в 2-3 предложениях своими словами;
   - почему это важно;
   - реакция сообщества (для HN — анализ комментариев, для HF — статистика лайков/скачиваний);
   - РЕЙТИНГ (1-10 баллов);
   - ТЕГИ для поиска (5-7 тегов, например: #ML #Security #NLP #Datasets и т.д.).

3. В конце добавь раздел «ГЛАВНЫЕ ВЫВОДЫ ДНЯ» — 3 ключевых тезиса.

ТРЕБОВАНИЯ К ФОРМАТУ (используй именно такую разметку):

🔝 ТОП НАХОДОК ДНЯ

1. [Заголовок]
Источник: [...] | Ссылка: [URL]
⭐ Рейтинг: [X/10]
🏷️ Теги: #tag1 #tag2 #tag3 #tag4 #tag5
Суть: [...]
Почему важно: [...]
Реакция сообщества: [...]

(и так далее по каждому материалу)

📌 ГЛАВНЫЕ ВЫВОДЫ ДНЯ
1. ...
2. ...
3. ...

Будь точен и критичен. Не выдумывай факты. Пиши живым, профессиональным русским языком."""

    print("\n🤖 Отправляю материалы в OpenRouter на анализ...")
    result = call_openrouter(prompt, max_tokens=3000)

    if result is None:
        return ("❌ Не удалось получить анализ: все модели недоступны.", [])

    return (result, all_materials)


# ============================================================================
# БЛОК 5: PODCAST-STYLE AUDIO (обсуждение, а не просто чтение)
# ============================================================================

def generate_podcast_script(digest: str, all_materials: List[Dict]) -> str:
    """
    Генерирует podcast-script на основе дайджеста.
    Вместо просто чтения текста — имитирует обсуждение двух ведущих.
    """
    materials_text = json.dumps(all_materials, ensure_ascii=False, indent=2)[:1000]

    prompt = f"""Ты — автор научного подкаста на русском языке. Твоя задача создать сценарий обсуждения \
между двумя ведущими (Алексей — опытный исследователь, Мария — аналитик) на основе следующего дайджеста.

ДАЙДЖЕСТ:
{digest[:1500]}

МАТЕРИАЛЫ:
{materials_text}

ТРЕБОВАНИЯ К СЦЕНАРИЮ:

1. Формат: диалог между Алексеем и Марией
2. Оба ведущих обсуждают ТОП-3 материала из дайджеста
3. Включи:
   - вступление (приветствие, тема выпуска);
   - для каждого материала: краткое описание, мнение Алексея, возражение/дополнение Марии;
   - рассуждения о тренде (что это означает, почему важно);
   - заключение с рекомендациями слушателям.

4. Длина: 4-5 минут чтения вслух (примерно 1500-2000 символов)
5. Язык: живой, естественный русский, как в реальном подкасте
6. Стиль: профессиональный но доступный

Структура:
[АЛЕКСЕЙ]: текст...
[МАРИЯ]: текст...

Начни сразу со сценария, без предисловий."""

    print("\n🎙️ Генерирую podcast-script...")
    script = call_openrouter(prompt, max_tokens=2500)
    
    if script is None:
        # Fallback: просто структурированный текст
        return f"""Добрый день, это научный дайджест за {datetime.now().strftime('%d.%m.%Y')}.

Сегодня мы рассмотрели материалы по информационной безопасности, машинному обучению и искусственному интеллекту.

Основные находки: {digest[:500]}

Слушайте наш полный дайджест в текстовой версии."""
    
    return script


def _prepare_podcast_tts_text(text: str) -> str:
    """Чистит podcast-скрипт для озвучки."""
    clean = (text
             .replace("[АЛЕКСЕЙ]:", "").replace("[МАРИЯ]:", "")
             .replace("**", "").replace("##", "")
             .replace("*", ""))
    # Добавляем паузы вместо смены ведущих
    clean = re.sub(r'\n\n+', '\n... \n', clean)
    intro = f"Подкаст: Научный дайджест за {datetime.now().strftime('%d.%m.%Y')}. "
    return intro + clean[:TTS_MAX_CHARS]


async def _synthesize_edge(speech_text: str, output_file: str) -> bool:
    """Microsoft Edge TTS."""
    try:
        import edge_tts
    except ImportError:
        print("   ⚠️  edge-tts не установлен", file=sys.stderr)
        return False

    try:
        communicate = edge_tts.Communicate(speech_text, TTS_VOICE)
        await communicate.save(output_file)
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"   ✓ Аудио (edge-tts) сохранено: {output_file}")
            return True
        print("   ⚠️  edge-tts вернул пустой файл", file=sys.stderr)
        return False
    except Exception as e:
        print(f"   ⚠️  edge-tts недоступен: {e}", file=sys.stderr)
        return False


def _synthesize_gtts(speech_text: str, output_file: str) -> bool:
    """Google TTS (запасной вариант)."""
    try:
        from gtts import gTTS
    except ImportError:
        print("   ⚠️  gTTS не установлен", file=sys.stderr)
        return False

    try:
        tts = gTTS(text=speech_text, lang="ru")
        tts.save(output_file)
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"   ✓ Аудио (gTTS) сохранено: {output_file}")
            return True
        return False
    except Exception as e:
        print(f"   ⚠️  gTTS недоступен: {e}", file=sys.stderr)
        return False


async def synthesize_podcast_audio(script: str, output_file: str = "podcast.mp3") -> Optional[str]:
    """
    Озвучивает podcast-сценарий (более естественный вариант, чем просто чтение).
    """
    speech_text = _prepare_podcast_tts_text(script)

    if await _synthesize_edge(speech_text, output_file):
        return output_file

    print("   → переключаюсь на запасной движок gTTS...")
    if _synthesize_gtts(speech_text, output_file):
        return output_file

    print("   ⚠️  Оба TTS-движка недоступны", file=sys.stderr)
    return None


# ============================================================================
# БЛОК 6: TELEGRAM ОТПРАВКА
# ============================================================================

def send_telegram_text(text: str) -> bool:
    """Отправляет текстовое сообщение в Telegram."""
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
        print(f"❌ Ошибка отправки текста: {e}", file=sys.stderr)
        return False


def send_telegram_audio(file_path: str) -> bool:
    """Отправляет аудиофайл в Telegram."""
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
                "title": f"Подкаст: Дайджест {datetime.now().strftime('%d.%m.%Y')}",
                "performer": "Science Digest Bot",
            }
            r = requests.post(url, data=payload, files=files, timeout=60)
            r.raise_for_status()
        print("✅ Аудио (подкаст) отправлено в Telegram")
        return True
    except Exception as e:
        print(f"❌ Ошибка отправки аудио: {e}", file=sys.stderr)
        return False


# ============================================================================
# БЛОК 7: METADATA И СОХРАНЕНИЕ АРХИВА
# ============================================================================

def save_digest_metadata(digest: str, materials: List[Dict], podcast_script: str) -> None:
    """Сохраняет метаданные дайджеста в JSON для анализа и архива."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "digest_text": digest,
        "materials_count": len(materials),
        "sources": {
            "arxiv": len([m for m in materials if m.get("source") == "arXiv"]),
            "hackernews": len([m for m in materials if m.get("source") == "Hacker News"]),
            "huggingface": len([m for m in materials if m.get("source") == "Hugging Face"]),
        },
        "materials": materials,
        "podcast_script": podcast_script,
    }
    
    filename = f"digest_{datetime.now().strftime('%Y%m%d')}.json"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"✅ Метаданные сохранены: {filename}")
    except Exception as e:
        print(f"⚠️  Не удалось сохранить метаданные: {e}", file=sys.stderr)


# ============================================================================
# БЛОК 8: ГЛАВНЫЙ WORKFLOW
# ============================================================================

def main():
    print(f"🚀 Science Digest Bot v2.0 (Podcast Edition) — {datetime.now().isoformat()}")
    print("-" * 80)

    # --- Шаг 1: сбор материалов ---
    print("📚 Сбор материалов...")
    
    print("  • arXiv...")
    arxiv_papers = fetch_arxiv_papers(["cs.CR", "cs.LG", "cs.AI"], max_results=6)
    print(f"    ✓ {len(arxiv_papers)} статей")

    print("  • Hacker News (истории + комментарии)...")
    hn_stories = fetch_hackernews_stories(max_stories=8, with_comments=True)
    print(f"    ✓ {len(hn_stories)} историй")

    print("  • Hugging Face (модели + датасеты)...")
    hf_items = fetch_huggingface_trending(max_items=6)
    print(f"    ✓ {len(hf_items)} моделей/датасетов")

    all_materials = arxiv_papers + hn_stories + hf_items
    print(f"\n📊 Всего материалов: {len(all_materials)}")
    print("-" * 80)

    if not all_materials:
        print("⚠️  Нет материалов, выходим.")
        send_telegram_text("⚠️ Сегодня не удалось собрать материалы для дайджеста.")
        return

    # --- Шаг 2: анализ и ранжирование ---
    digest, materials_with_metadata = analyze_materials(all_materials)

    print("\n" + "=" * 80)
    print("📰 ДАЙДЖЕСТ:")
    print("=" * 80)
    print(digest)
    print("=" * 80)

    # --- Шаг 3: генерируем podcast-скрипт ---
    podcast_script = generate_podcast_script(digest, all_materials)

    # --- Шаг 4: формируем итоговое сообщение ---
    full_message = (
        f"🔬 НАУЧНЫЙ ДАЙДЖЕСТ\n"
        f"{datetime.now().strftime('%d.%m.%Y %H:%M')}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{digest}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Источники: arXiv ({len(arxiv_papers)}), "
        f"Hacker News ({len(hn_stories)}), "
        f"Hugging Face ({len(hf_items)})\n\n"
        f"🤖 Science Digest Bot v2.0"
    )

    # --- Шаг 5: отправляем текст ---
    print("\n📤 Отправляю текст в Telegram...")
    send_telegram_text(full_message)

    # --- Шаг 6: озвучка подкаста и отправка аудио ---
    if ENABLE_AUDIO:
        print("\n🎙️  Генерирую podcast-аудио...")
        audio_file = asyncio.run(synthesize_podcast_audio(podcast_script))
        if audio_file:
            send_telegram_audio(audio_file)

    # --- Шаг 7: сохраняем метаданные ---
    print("\n💾 Сохраняю метаданные...")
    save_digest_metadata(digest, all_materials, podcast_script)

    print("\n✅ Готово!")


if __name__ == "__main__":
    main()
