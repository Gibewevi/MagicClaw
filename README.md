# Magic Claw

Magic Claw is an independent local terminal agent designed for long-running
operation. It keeps the architecture intentionally small:

- hardware diagnosis
- automatic model recommendation
- GGUF model download
- llama.cpp server supervision
- SQLite progress memory
- local file and shell tools
- optional Telegram control

## Quick start

Windows:

```powershell
.\scripts\run.ps1
```

Manual:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -m magic_claw
```

## Commands

```powershell
python -m magic_claw              # interactive plug and play startup
python -m magic_claw diagnose     # hardware report
python -m magic_claw models       # search recent GGUF models and show compatible choices
python -m magic_claw init         # choose automatic or manual model selection
python -m magic_claw run          # start supervisor
python -m magic_claw task "..."   # run one local agent task
```

During `init`, Magic Claw first displays `Recherche des modèles récents...`,
queries Hugging Face for recent GGUF models, filters them against the detected
GPU/RAM profile, then offers:

- Automatic: selects the strongest stable model for the machine.
- Manual: displays the compatible/recommended model list.

Telegram can be enabled during setup without editing files:

```powershell
python -m magic_claw init --telegram-token "123:abc" --telegram-user-id 123456789
```

The token is stored in `.magicclaw/.env`, not in the generated JSON config.

## Stability rules

- one state database: `.magicclaw/state/magic_claw.sqlite`
- one generated config: `.magicclaw/config.generated.json`
- no runtime edits to source files
- model process is isolated from the agent process
- every tool call is checkpointed before the next step
- shell calls always have timeouts
