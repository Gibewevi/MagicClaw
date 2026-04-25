from magic_claw.cli import InteractivePromptBuffer


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
