import re


def fix_markdown_symbol_issue(md: str) -> str:
    """
    Fixes common markdown symbol issues for Streamlit display.
    - Escapes $ followed by digits.
    - Escapes ~ to prevent unintended strikethrough.
    - Adds spacing around bold/italic markers if they contain special characters or quotes,
      which often breaks rendering in some markdown parsers.
    - Ignores content inside code blocks.
    """
    # Pattern to find code blocks (triple backticks or single backtick)
    # We want to exclude these from symbol escaping
    # Captures: 1. Triple backticks blocks, 2. Inline code (simple `...`)
    code_block_pattern = r"(```[\s\S]*?```|`[^`]*`)"

    parts = re.split(code_block_pattern, md)

    # Clean up ** surrounding code blocks (odd indices in parts)
    for i in range(1, len(parts), 2):
        if parts[i - 1].endswith("**") and parts[i + 1].startswith("**"):
            parts[i - 1] = parts[i - 1][:-2]
            parts[i + 1] = parts[i + 1][2:]

    # Pattern for the bold fix
    bold_pattern = re.compile(r"\*\*(.+?)\*\*(\s*)")

    def bold_repl(m):
        inner = m.group(1)
        after = m.group(2)
        inner = inner.strip()
        # Add space after ** if content contains symbols and no space exists
        # This helps when bold text is immediately followed by punctuation or other text
        # that might confuse the renderer if not separated.
        if re.search(r"[^0-9A-Za-z\s가-힣]", inner) and after == "":
            return f"**{inner}** "
        if inner != m.group(1):
            return f"**{inner}**{after}"
        return m.group(0)

    # Pattern for the italic fix (avoid matching bold **)
    italic_pattern = re.compile(r"(?<!\*)\*(?![*])(.+?)(?<!\*)\*(?![*])(\s*)")

    def italic_repl(m):
        inner = m.group(1)
        after = m.group(2)
        # Add space after * if content contains quotes and no space exists
        if re.search(r"['\"]", inner) and after == "":
            return f"*{inner}* "
        return m.group(0)

    for i in range(len(parts)):
        # Even indices are regular text; Odd indices are code blocks (the delimiters)
        if i % 2 == 0:
            part = parts[i]

            # 1. Escape $ only if followed by a digit (e.g. $100)
            # This prevents Streamlit/KaTeX from interpreting it as LaTeX math.
            part = re.sub(r"\$(\d)", r"\\$\1", part)

            # 2. Escape ~ to prevent strikethrough interpretation
            part = part.replace("~", "\\~")

            # 3. Apply bold spacing fix
            part = bold_pattern.sub(bold_repl, part)

            # 4. Apply italic spacing fix
            part = italic_pattern.sub(italic_repl, part)

            parts[i] = part

    return "".join(parts)
