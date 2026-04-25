from magic_claw.cli import InteractivePromptBuffer, _with_interactive_context


def test_prompt_buffer_runs_normal_requests_immediately():
    buffer = InteractivePromptBuffer()

    prompt, from_buffer = buffer.add("Creer un site meteo avec Vite")

    assert prompt == "Creer un site meteo avec Vite"
    assert from_buffer is False


def test_prompt_buffer_combines_section_heading_with_next_line():
    buffer = InteractivePromptBuffer()

    prompt, from_buffer = buffer.add("Objectif :")
    assert prompt is None
    assert from_buffer is True

    prompt, from_buffer = buffer.add("Creer un site meteo avec Vite")
    assert prompt == "Objectif :\nCreer un site meteo avec Vite"
    assert from_buffer is True


def test_interactive_context_is_bounded():
    history = [("old prompt " + ("p" * 5000), "old result " + ("r" * 5000)) for _ in range(5)]

    prompt = _with_interactive_context("current request", history)

    assert "current request" in prompt
    assert prompt.count("Previous user request") == 3
    assert len(prompt) < 10000
    assert "historique tronqué" in prompt
