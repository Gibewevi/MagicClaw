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
python -m magic_claw telegram     # show Telegram control status
python -m magic_claw telegram setup
python -m magic_claw run          # start supervisor
python -m magic_claw task "..."   # run one local agent task
```

During `init`, Magic Claw first displays `Recherche des modèles récents...`,
queries Hugging Face for recent GGUF models, filters them against the detected
GPU/RAM profile, then offers:

- Automatic: selects the strongest stable model for the machine.
- Manual: displays the compatible/recommended model list.

On first setup, Magic Claw asks whether you want Telegram control after model
selection. If enabled, paste the bot token created with Telegram's BotFather.
Magic Claw validates the token with Telegram before saving it.

Telegram can also be configured later without editing files:

```powershell
python -m magic_claw telegram setup
python -m magic_claw telegram status
python -m magic_claw telegram disable
python -m magic_claw telegram reset
```

The token is stored in `.magicclaw/.env`; enabled/disabled state and bot metadata
are stored in `.magicclaw/config.generated.json`. When configured, `run` starts
Telegram polling and the terminal prompt at the same time. In Telegram, send
`/help`, `/status`, or any task message.

## Stability rules

- one state database: `.magicclaw/state/magic_claw.sqlite`
- one generated config: `.magicclaw/config.generated.json`
- no runtime edits to source files
- model process is isolated from the agent process
- every tool call is checkpointed before the next step
- reaching the step window creates a durable continuation memory and resumes
- shell calls always have absolute timeouts; npm-like commands also have an
  inactivity watchdog and process-tree termination
