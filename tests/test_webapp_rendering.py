from __future__ import annotations

from webapp import _message_to_response, _render_assistant_markdown


def test_renders_standard_assistant_markdown():
    rendered = _render_assistant_markdown(
        "# Heading\n\nA **bold** and *emphasized* paragraph with [a link](https://example.com/a).\n\n"
        "1. First\n2. Second\n\n- Parent\n  - Child\n\n`inline`\n\n```python\nprint('ok')\n```"
    )

    assert "<h1>Heading</h1>" in rendered
    assert "<strong>bold</strong>" in rendered
    assert "<em>emphasized</em>" in rendered
    assert '<a href="https://example.com/a" rel="noopener noreferrer">a link</a>' in rendered
    assert "<ol>" in rendered and "<ul>" in rendered
    assert "<code>inline</code>" in rendered
    assert '<code class="language-python">' in rendered


def test_sanitizes_html_scripts_handlers_and_unsafe_urls():
    rendered = _render_assistant_markdown(
        '<script>alert(1)</script>\n\n<img src=x onerror="alert(2)">\n\n'
        '[unsafe](javascript:alert(3)) [data](data:text/html,bad) [safe](mailto:test@example.com)'
    )

    assert "<script" not in rendered
    assert "<img" not in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "&lt;img src=x onerror=\"alert(2)\"&gt;" in rendered
    assert 'href="javascript:' not in rendered
    assert 'href="data:' not in rendered
    assert 'href="mailto:test@example.com"' in rendered


def test_plain_and_incomplete_markdown_remains_readable():
    rendered = _render_assistant_markdown("Plain response\n\n**unfinished")

    assert "<p>Plain response</p>" in rendered
    assert "**unfinished" in rendered


def test_only_assistant_response_gets_additive_rendered_content():
    assistant = _message_to_response({"role": "assistant", "content": "**bold**", "id": 1})
    user = _message_to_response({"role": "user", "content": "**literal**", "id": 2})

    assert assistant["content"] == "**bold**"
    assert assistant["rendered_content"] == "<p><strong>bold</strong></p>\n"
    assert "rendered_content" not in user
    assert user["content"] == "**literal**"
