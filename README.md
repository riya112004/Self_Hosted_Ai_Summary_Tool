## Local low-memory setup

For a machine with around 4 GB of RAM, run one FastAPI worker and use the
smaller Ollama model configured in `.env`:

```powershell
ollama pull qwen3:0.6b
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Do not start Uvicorn with multiple workers on a 4 GB machine. Each worker can
load its own application state while Ollama also needs memory for the model.
The smaller model is slower or less capable than `qwen3:1.7b`, but leaves more
RAM available for FastAPI and document processing.

### Windows swap/pagefile

If RAM pressure causes failures, increase the Windows pagefile rather than
trying to solve it with more application workers:

1. Open **System Properties** and choose **Advanced** > **Performance Settings**.
2. Open **Advanced** > **Virtual memory** > **Change**.
3. Enable **Automatically manage paging file size**, or set a custom size on a
	drive with free space (for example, 4096 MB initial and 8192 MB maximum).
4. Restart Windows, then start Ollama and FastAPI with one worker.

Pagefile space prevents out-of-memory crashes but is much slower than RAM, so
it does not make inference faster.

Document map-reduce is sequential by default to keep memory stable. The
configured 500-1500 word range reduces unnecessary LLM calls while avoiding
very large prompts. Only add batching or concurrency after confirming the
machine has enough RAM for multiple Ollama requests.
