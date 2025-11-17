# Ollama Integration - Quick Reference Guide

**For developers implementing the Ollama integration**

---

## Essential Models

### Download Commands

```bash
# MINIMAL SETUP (8GB RAM)
ollama pull qwen2.5:7b          # 4.7GB - Fast tasks
ollama pull llama3.2:3b         # 2GB - Quick checks

# RECOMMENDED SETUP (16GB RAM)
ollama pull aya-expanse:8b      # ~5GB - Hebrew expert ⭐
ollama pull llama3.2:8b         # ~5GB - Good reasoning
ollama pull qwen2.5:7b          # 4.7GB - Fast extraction

# POWER SETUP (32GB+ RAM, GPU)
ollama pull aya-expanse:8b      # ~5GB - Hebrew expert ⭐
ollama pull llama3.2:70b        # ~40GB - Best reasoning
ollama pull qwen2.5:7b          # 4.7GB - Fast extraction
ollama pull llama3.2:3b         # 2GB - Quick tasks
```

---

## API Quick Reference

### Check if Ollama is Running

```python
import requests

def is_ollama_available():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False
```

### List Available Models

```python
def get_available_models():
    response = requests.get("http://localhost:11434/api/tags")
    return [m['name'] for m in response.json()['models']]

# Example output:
# ['aya-expanse:8b', 'llama3.2:8b', 'qwen2.5:7b']
```

### Generate Text (Non-Streaming)

```python
def generate_text(model, prompt, temperature=0.7):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 40
            }
        },
        timeout=120
    )
    return response.json()['response']

# Usage
result = generate_text("aya-expanse:8b", "סכם את הטקסט הזה: ...")
```

### Generate with Streaming

```python
def generate_streaming(model, prompt, callback=None):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )

    accumulated = ""
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            text = chunk.get('response', '')
            accumulated += text

            if callback:
                callback(text)  # Real-time UI update

            if chunk.get('done', False):
                break

    return accumulated

# Usage with callback
def update_ui(chunk):
    print(chunk, end='', flush=True)

result = generate_streaming("aya-expanse:8b", "Summarize: ...", update_ui)
```

---

## Common Prompts

### Summary Generation

```python
def generate_summary(transcript, model="aya-expanse:8b"):
    prompt = f"""Analyze this Hebrew/English video transcript and create a summary.

Provide:
1. TL;DR (one sentence)
2. Paragraph summary (2-3 sentences)
3. Detailed summary (5-8 sentences)

Transcript:
{transcript[:3000]}

Respond in JSON format:
{{
    "tldr": "...",
    "paragraph": "...",
    "detailed": "..."
}}"""

    response = generate_text(model, prompt, temperature=0.3)
    return parse_json_response(response)
```

### Topic Extraction

```python
def extract_topics(transcript, model="qwen2.5:7b"):
    prompt = f"""Extract main topics from this transcript.

For each topic provide:
- Name
- Relevance score (0-1)
- Key subtopics

Transcript:
{transcript[:2000]}

Respond in JSON:
{{
    "topics": [
        {{
            "name": "Topic Name",
            "relevance": 0.92,
            "subtopics": ["Sub1", "Sub2"]
        }}
    ]
}}"""

    response = generate_text(model, prompt, temperature=0.2)
    return parse_json_response(response)
```

### Reel Segment Detection

```python
def detect_reel_segments(transcript, segments, model="llama3.2:70b"):
    # segments = list of {start, end, text} from transcription

    segments_text = "\n".join([
        f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}"
        for seg in segments[:50]  # First 50 segments only
    ])

    prompt = f"""Identify 3-5 video segments (15-60 seconds) that would work well as social media Reels/Shorts.

Criteria:
- Engaging hook (first 3 seconds)
- Self-contained (makes sense alone)
- Interesting or educational
- High energy or emotional impact

Transcript with timestamps:
{segments_text}

Respond in JSON:
{{
    "segments": [
        {{
            "start": 45.2,
            "end": 62.8,
            "duration": 17.6,
            "score": 94,
            "reason": "Strong hook with practical demo"
        }}
    ]
}}"""

    response = generate_text(model, prompt, temperature=0.4)
    return parse_json_response(response)
```

### Title Generation

```python
def generate_titles(transcript, count=10, model="llama3.2:8b"):
    prompt = f"""Generate {count} engaging video titles in Hebrew and English for this content.

Transcript summary:
{transcript[:1000]}

Create a mix of:
- Curiosity-driven ("You Won't Believe...")
- Benefit-focused ("Learn X in Y Minutes")
- Question-based ("Is This Really True?")
- Listicle ("7 Secrets About...")

Respond in JSON:
{{
    "titles": [
        {{
            "title": "למד X ב-5 דקות",
            "type": "benefit-focused",
            "language": "hebrew"
        }}
    ]
}}"""

    response = generate_text(model, prompt, temperature=0.8)  # Higher temp for creativity
    return parse_json_response(response)
```

### Hashtag Generation

```python
def generate_hashtags(transcript, model="aya-expanse:8b"):
    prompt = f"""Generate relevant hashtags for this Hebrew/English video content.

Transcript:
{transcript[:1000]}

Provide:
- 3-5 broad hashtags (high volume)
- 5-7 medium hashtags (targeted)
- 2-4 niche hashtags (specific community)

Include both Hebrew and English tags.

Respond in JSON:
{{
    "broad": ["#AI", "#MachineLearning"],
    "medium": ["#PythonProgramming", "#DataScience"],
    "niche": ["#למידתמכונה", "#HebrewTech"]
}}"""

    response = generate_text(model, prompt, temperature=0.5)
    return parse_json_response(response)
```

---

## Multi-Model Patterns

### Sequential Chaining (Quality Focus)

```python
def analyze_with_chaining(transcript):
    # Step 1: Fast extraction (Qwen)
    candidates = detect_reel_segments(transcript, model="qwen2.5:7b")

    # Step 2: Deep analysis (Llama)
    scored_segments = []
    for candidate in candidates['segments']:
        score = analyze_segment_quality(
            candidate,
            model="llama3.2:70b"
        )
        scored_segments.append({**candidate, **score})

    # Step 3: Hebrew validation (Aya)
    final_segments = []
    for segment in scored_segments:
        validation = validate_hebrew_context(
            segment,
            model="aya-expanse:8b"
        )
        if validation['approved']:
            final_segments.append(segment)

    return final_segments
```

### Parallel Processing (Speed Focus)

```python
from concurrent.futures import ThreadPoolExecutor

def analyze_parallel(transcript):
    tasks = {
        'summary': lambda: generate_summary(transcript),
        'topics': lambda: extract_topics(transcript),
        'hashtags': lambda: generate_hashtags(transcript),
        'titles': lambda: generate_titles(transcript),
        'reels': lambda: detect_reel_segments(transcript)
    }

    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(task_fn): task_name
            for task_name, task_fn in tasks.items()
        }

        for future in concurrent.futures.as_completed(futures):
            task_name = futures[future]
            try:
                results[task_name] = future.result(timeout=300)
                print(f"✅ {task_name} completed")
            except Exception as e:
                print(f"❌ {task_name} failed: {e}")
                results[task_name] = {"error": str(e)}

    return results
```

---

## Helper Functions

### Parse JSON from LLM Response

```python
import re
import json

def parse_json_response(response):
    """Extract JSON from LLM response (handles markdown code blocks)"""
    # Try to find JSON object in response
    json_match = re.search(r'\{.*\}', response, re.DOTALL)

    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: return raw response
    return {"raw_response": response, "parse_error": True}
```

### Build Context-Aware Prompt

```python
def build_prompt(template, transcript, metadata):
    """Build prompt with video metadata context"""

    context = f"""
Video Metadata:
- Duration: {metadata['duration']:.1f} seconds
- Language: {metadata.get('language', 'unknown')}
- Segment count: {len(metadata.get('segments', []))}

{template}
"""

    # Truncate transcript if too long
    max_length = 3000
    if len(transcript) > max_length:
        transcript = transcript[:max_length] + "\n\n[...truncated...]"

    return context.replace("{transcript}", transcript)
```

---

## Error Handling

### Graceful Degradation

```python
class OllamaAnalyzer:
    def __init__(self):
        self.available = self.check_availability()

    def check_availability(self):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def analyze(self, transcript):
        if not self.available:
            print("⚠️  Ollama not available. Skipping AI analysis.")
            print("   Install from: https://ollama.ai")
            return None

        try:
            result = self._run_analysis(transcript)
            return result
        except Exception as e:
            print(f"⚠️  AI analysis failed: {e}")
            return None

    def _run_analysis(self, transcript):
        # Actual analysis implementation
        pass
```

### Retry Logic

```python
import time

def generate_with_retry(model, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return generate_text(model, prompt)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # Exponential backoff
                print(f"⚠️  Attempt {attempt+1} failed, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise e
```

---

## Performance Optimization

### Caching

```python
import hashlib
import json
from pathlib import Path

class AnalysisCache:
    def __init__(self, cache_dir="~/.reels_extractor/cache"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_key(self, transcript, analysis_type, model):
        content = f"{transcript}_{analysis_type}_{model}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, cache_key):
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                cached = json.load(f)
                # Check if cache is less than 7 days old
                import time
                age = time.time() - cached['timestamp']
                if age < 7 * 24 * 3600:  # 7 days
                    return cached['result']
        return None

    def set(self, cache_key, result):
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump({
                'result': result,
                'timestamp': time.time()
            }, f)

# Usage
cache = AnalysisCache()
cache_key = cache.get_key(transcript, "summary", "aya-expanse:8b")

result = cache.get(cache_key)
if not result:
    result = generate_summary(transcript)
    cache.set(cache_key, result)
```

### Transcript Truncation

```python
def truncate_transcript(transcript, max_tokens=3000, strategy="smart"):
    """
    Truncate transcript intelligently based on strategy

    Strategies:
    - 'start': First N tokens
    - 'end': Last N tokens
    - 'smart': First half + last half (for summary)
    - 'full': Don't truncate (chunk instead)
    """

    if len(transcript) <= max_tokens:
        return transcript

    if strategy == "start":
        return transcript[:max_tokens]

    elif strategy == "end":
        return transcript[-max_tokens:]

    elif strategy == "smart":
        half = max_tokens // 2
        return transcript[:half] + "\n\n[...truncated...]\n\n" + transcript[-half:]

    elif strategy == "full":
        # Return list of chunks
        chunks = []
        for i in range(0, len(transcript), max_tokens):
            chunks.append(transcript[i:i+max_tokens])
        return chunks

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
```

---

## Integration with Existing Code

### Modify transcribe_advanced.py

```python
# At the end of transcribe_video() function:

def transcribe_video(video_path):
    # ... existing transcription code ...

    # NEW: AI Analysis with Ollama
    from ollama_analyzer import OllamaAnalyzer

    analyzer = OllamaAnalyzer()
    if analyzer.available:
        print("\n🤖 Running AI analysis with Ollama...")

        # Run parallel analysis
        ai_results = analyzer.analyze_parallel(result['text'], result['segments'])

        # Save AI analysis
        save_ai_analysis(output_dir, ai_results)

        # Display results
        display_ai_results(ai_results)

        print("✅ AI analysis complete")
    else:
        print("⚠️  Ollama not running. Skipping AI analysis.")
        print("   Install from: https://ollama.ai")

    return result
```

### Output Structure

```python
def save_ai_analysis(output_dir, ai_results):
    """Save AI analysis results in multiple formats"""

    # JSON (machine-readable)
    json_path = os.path.join(output_dir, "ai_analysis.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(ai_results, f, ensure_ascii=False, indent=2)

    # Markdown (human-readable)
    md_path = os.path.join(output_dir, "ai_analysis.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(format_as_markdown(ai_results))

    # HTML (rich display)
    html_path = os.path.join(output_dir, "ai_analysis.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(format_as_html(ai_results))

    print(f"📄 AI analysis saved:")
    print(f"   JSON: {json_path}")
    print(f"   Markdown: {md_path}")
    print(f"   HTML: {html_path}")
```

---

## Testing

### Test Model Availability

```python
def test_ollama_setup():
    """Verify Ollama is installed and models are available"""

    print("🔍 Checking Ollama setup...\n")

    # Check if Ollama is running
    if not is_ollama_available():
        print("❌ Ollama is not running")
        print("   Start it with: ollama serve")
        return False

    print("✅ Ollama is running")

    # Check available models
    models = get_available_models()
    print(f"\n📦 Installed models ({len(models)}):")
    for model in models:
        print(f"   - {model}")

    # Check recommended models
    recommended = ["aya-expanse:8b", "qwen2.5:7b", "llama3.2:8b"]
    missing = [m for m in recommended if m not in models]

    if missing:
        print(f"\n⚠️  Missing recommended models:")
        for model in missing:
            print(f"   ollama pull {model}")

    return len(missing) == 0

# Run test
test_ollama_setup()
```

### Test Analysis Functions

```python
def test_analysis():
    """Test AI analysis with sample transcript"""

    sample_transcript = """
    שלום לכולם, היום נדבר על למידת מכונה.
    Machine learning is a fascinating field.
    נתחיל עם supervised learning...
    """

    print("🧪 Testing AI analysis functions...\n")

    # Test summary
    print("1. Testing summary generation...")
    summary = generate_summary(sample_transcript)
    print(f"   ✅ Summary: {summary.get('tldr', 'N/A')}\n")

    # Test topics
    print("2. Testing topic extraction...")
    topics = extract_topics(sample_transcript)
    print(f"   ✅ Topics: {topics.get('topics', [])}\n")

    # Test hashtags
    print("3. Testing hashtag generation...")
    hashtags = generate_hashtags(sample_transcript)
    print(f"   ✅ Hashtags: {hashtags}\n")

    print("All tests passed! ✅")

# Run tests
test_analysis()
```

---

## Common Issues & Solutions

### Issue: "Connection refused"
**Solution**: Ollama server not running
```bash
# Start Ollama server
ollama serve
```

### Issue: "Model not found"
**Solution**: Model not downloaded
```bash
# Pull the model
ollama pull aya-expanse:8b
```

### Issue: "Out of memory"
**Solution**: Model too large for RAM
```bash
# Use smaller model
ollama pull llama3.2:3b  # Instead of :70b
```

### Issue: "Response too slow"
**Solution**: Use faster model or GPU
```bash
# Switch to faster model
ollama pull qwen2.5:7b

# Or enable GPU (if available)
# Ollama automatically uses GPU if detected
```

### Issue: "JSON parsing failed"
**Solution**: LLM response not valid JSON
```python
# Add retry with explicit JSON instruction
prompt += "\n\nIMPORTANT: Respond with valid JSON only, no markdown."
```

---

## CLI Flags (After Implementation)

```bash
# Basic transcription (existing)
python transcribe_advanced.py

# With AI analysis (new)
python transcribe_advanced.py --ai-analysis

# Specify model
python transcribe_advanced.py --ai-analysis --ollama-model aya-expanse:8b

# Skip cache
python transcribe_advanced.py --ai-analysis --no-cache

# Parallel processing
python transcribe_advanced.py --ai-analysis --parallel --max-workers 4

# Export formats
python transcribe_advanced.py --ai-analysis --export json,markdown,html
```

---

## Resources

- **Ollama Documentation**: https://github.com/ollama/ollama/blob/main/docs/api.md
- **Aya-Expanse Model**: https://ollama.com/library/aya-expanse
- **Llama3.2**: https://ollama.com/library/llama3.2
- **Qwen2.5**: https://ollama.com/library/qwen2.5

---

**Quick Links**:
- Full Proposal: `OLLAMA_ADVANCED_INTEGRATION_PROPOSAL.md`
- Summary: `OLLAMA_INTEGRATION_SUMMARY.md`
- Project README: `../README.md`

---

**Last Updated**: 2025-10-21
