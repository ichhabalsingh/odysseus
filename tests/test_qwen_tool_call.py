import src.agent_tools  # noqa: F401
from src.tool_parsing import parse_tool_blocks, strip_tool_blocks


def test_qwen_tool_call_parsing_and_stripping():
    raw = """Sure, let me check your accounts.

<|tool_call_start|>[list_email_accounts()]<|tool_call_end|>"""

    blocks = parse_tool_blocks(raw, skip_fenced=True)

    assert len(blocks) == 1
    assert blocks[0].tool_type == "mcp__email__list_email_accounts"
    assert blocks[0].content == "{}"
    assert strip_tool_blocks(raw, skip_fenced=True) == "Sure, let me check your accounts."


def test_qwen_tool_call_with_args():
    raw = """Okay, fetching recent messages.
<|tool_call_start|>[list_emails(account="Gmail", unread_only=True, max_results=5)]<|tool_call_end|>"""

    blocks = parse_tool_blocks(raw, skip_fenced=True)

    assert len(blocks) == 1
    assert blocks[0].tool_type == "mcp__email__list_emails"
    assert "Gmail" in blocks[0].content
    
    cleaned = strip_tool_blocks(raw, skip_fenced=True)
    assert cleaned == "Okay, fetching recent messages."


def test_qwen_tool_call_positional_args():
    # Single positional argument
    raw = '<|tool_call_start|>[web_search("Sweden news")]<|tool_call_end|>'
    blocks = parse_tool_blocks(raw, skip_fenced=True)
    assert len(blocks) == 1
    assert blocks[0].tool_type == "web_search"
    assert "Sweden news" in blocks[0].content

    # Multiple positional arguments
    raw = '<|tool_call_start|>[read_file("src/main.py", 10, 50)]<|tool_call_end|>'
    blocks = parse_tool_blocks(raw, skip_fenced=True)
    assert len(blocks) == 1
    assert blocks[0].tool_type == "read_file"
    assert "src/main.py" in blocks[0].content
    assert "10" in blocks[0].content
    assert "50" in blocks[0].content


def test_qwen_tool_call_whitespace_before_end_tag():
    raw = "Okay.\n<|tool_call_start|>[web_search(query=\"Sweden news\")]\n<|tool_call_end|>"
    blocks = parse_tool_blocks(raw, skip_fenced=True)
    assert len(blocks) == 1
    assert blocks[0].tool_type == "web_search"
    
    cleaned = strip_tool_blocks(raw, skip_fenced=True)
    assert cleaned == "Okay."


def test_qwen_tool_call_regex_fallback_and_single_arg():
    # Regex fallback with unquoted values containing spaces
    raw = '<|tool_call_start|>[web_search(query=Sweden news today, time_filter=day)]<|tool_call_end|>'
    blocks = parse_tool_blocks(raw, skip_fenced=True)
    assert len(blocks) == 1
    assert blocks[0].tool_type == "web_search"
    assert "Sweden news today" in blocks[0].content
    assert "day" in blocks[0].content

    # Single argument fallback (syntax error, no keyword)
    raw = '<|tool_call_start|>[web_search(Sweden news today)]<|tool_call_end|>'
    blocks = parse_tool_blocks(raw, skip_fenced=True)
    assert len(blocks) == 1
    assert blocks[0].tool_type == "web_search"
    assert "Sweden news today" in blocks[0].content


def test_qwen_argument_validation():
    # list-valued command in bash
    raw = '<|tool_call_start|>[bash(command=["ls", "-la"])]<|tool_call_end|>'
    blocks = parse_tool_blocks(raw, skip_fenced=True)
    assert len(blocks) == 0

    # set-valued query in web_search
    raw = '<|tool_call_start|>[web_search(query={"Sweden news"})]<|tool_call_end|>'
    blocks = parse_tool_blocks(raw, skip_fenced=True)
    assert len(blocks) == 0

    # non-string path/content in write_file
    raw = '<|tool_call_start|>[write_file(path=123, content=["foo"])]<|tool_call_end|>'
    blocks = parse_tool_blocks(raw, skip_fenced=True)
    assert len(blocks) == 0


def test_qwen_positional_canonicalization():
    # Alias shell maps to bash, and its positional argument maps to command
    raw = '<|tool_call_start|>[shell("pwd")]<|tool_call_end|>'
    blocks = parse_tool_blocks(raw, skip_fenced=True)
    assert len(blocks) == 1
    assert blocks[0].tool_type == "bash"
    assert blocks[0].content == "pwd"


def test_qwen_email_fail_closed():
    # Positional list_emails call should fail closed (return None/no blocks)
    raw = '<|tool_call_start|>[list_emails("work")]<|tool_call_end|>'
    blocks = parse_tool_blocks(raw, skip_fenced=True)
    assert len(blocks) == 0

    # Keyword list_emails call should parse successfully
    raw = '<|tool_call_start|>[list_emails(account="work")]<|tool_call_end|>'
    blocks = parse_tool_blocks(raw, skip_fenced=True)
    assert len(blocks) == 1
    assert blocks[0].tool_type == "mcp__email__list_emails"
    assert "work" in blocks[0].content


def test_qwen_delimiter_parsing():
    # )] sequence inside quotes should be treated as data
    raw = '<|tool_call_start|>[web_search(query="a )] b")]<|tool_call_end|>'
    blocks = parse_tool_blocks(raw, skip_fenced=True)
    assert len(blocks) == 1
    assert blocks[0].tool_type == "web_search"
    assert "a )] b" in blocks[0].content

    # Should also strip correctly
    cleaned = strip_tool_blocks(raw, skip_fenced=True)
    assert cleaned == ""


def test_qwen_parser_robustness_incomplete_wrapper():
    # Incomplete wrapper (no closing tag) should not execute or strip
    raw = 'before <|tool_call_start|>[bash("id")] after'
    blocks = parse_tool_blocks(raw, skip_fenced=True)
    assert len(blocks) == 0

    cleaned = strip_tool_blocks(raw, skip_fenced=True)
    assert cleaned == 'before <|tool_call_start|>[bash("id")] after'


def test_qwen_email_malformed_keyword_fail_closed():
    # If key-value evaluation fails for email tools, it should fail closed (return no blocks)
    raw = '<|tool_call_start|>[list_emails(account=work)]<|tool_call_end|>'
    blocks = parse_tool_blocks(raw, skip_fenced=True)
    assert len(blocks) == 0
