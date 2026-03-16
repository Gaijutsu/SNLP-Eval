"""
Python file that gatherers can use to get their prompts
"""

# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------

AGENTLESS_FILE_LOCALIZATION_PROMPT = """\
You are an expert software engineer performing fault localization on a code repository.
- Sort candidate files by relevance, with the most relevant first. Relevance is based on how strongly the file is connected to the issue through identifiers, directory structure, and content.
- You must always provide the test files for the relevant code, if they exist, as they often contain crucial information about how the code is used and what the expected behavior is.


## Issue
{query}

## Repository Files
{file_listing}

## Task
Identify which files are most relevant to this issue. Reason step by step:

1. Extract key identifiers from the issue: function/class names, error messages,
   feature keywords, and module names that appear in the issue description.
2. Match those identifiers against the file listing using path names and directory
   structure (e.g. an auth bug likely lives under `auth/` or `login`).
3. Consider indirect relevance: configuration files, shared utilities, and test
   files that the primary file depends on or exercises.

Respond with a JSON array of up to {top_n} file paths, ranked by relevance
(most relevant first). Output ONLY the JSON array, with no surrounding text.

Example: ["src/auth/login.py", "tests/test_auth.py", "src/models/user.py"]

/nothink
"""

AGENTLESS_FUNCTION_LOCALIZATION_PROMPT = """\
You are an expert software engineer. Given the issue and the retrieved file
contents below, identify the precise code regions that need to be modified
to fix this issue.

## Issue
{query}

## File Contents
{file_contents}

## Task
For each file that requires changes, pinpoint the exact code region:
- The function, method, or class name involved
- An approximate line range (e.g. "42-78")
- A brief reason explaining why this region is responsible for the issue

Respond with a JSON array. Each element must have:
- "file": the file path
- "symbol": the function, method, or class name (use "module-level" for top-level code)
- "line_range": approximate line range as a string, e.g. "42-78"
- "reason": one sentence explaining why this region needs to change

Output ONLY the JSON array, with no surrounding text.

Example: [{{"file": "src/auth.py", "symbol": "login", "line_range": "42-65", "reason": "Credential comparison is case-sensitive when it should be case-insensitive."}}]

/nothink
"""

AGENTLESS_REPAIR_PROMPT = """\
You are an expert software engineer. Your task is to generate a minimal,
correct patch in unified diff format to fix the issue described below.

## Issue
{query}

## Relevant Code
{code_regions}

## Instructions
Think step by step:
1. Identify the root cause of the bug in the code above.
2. Determine the minimal change needed to fix it.
3. Consider edge cases (e.g. None inputs, empty collections, type mismatches).

Then generate the patch. Rules:
- Output a single unified diff block, starting with ```diff and ending with ```.
- Only change lines directly related to the fix; do not reformat unrelated code.
- Preserve the existing code style (indentation, naming conventions).
- If multiple files need changes, include all hunks in one diff block.
- The patch must actually fix the described bug, not just reformat comments or docstrings.
"""


REACT_TOOL_DESCRIPTIONS = """\
You have the following tools available:

1. list_dir(path: str) -> str
   List files and subdirectories in the given directory (relative to repo root).
   Use this to understand project layout around a candidate directory.
   Example: list_dir("src/auth")

2. grep(pattern: str, path: str = ".") -> str
   Search for a regex pattern in source files (.py/.java/.ts/.js/.cs) under the
   given path. Returns matching lines with file:line format. Use this to find
   where a specific function, class, or error message is defined or called.
   Example: grep("def authenticate", "src/")

3. read_file(path: str, start_line: int = 1, end_line: int = 200) -> str
   Read the contents of a file (relative to repo root). Use this to confirm a
   file's relevance by inspecting its implementation. You may optionally pass a
   start and end line range when grep already showed the interesting region.
   Example: read_file("src/auth/login.py")
   Example: read_file("src/auth/login.py", 120, 220)

/nothink
"""

REACT_SYSTEM_PROMPT = """\
You are a code investigation agent. Your goal is to identify the files most
relevant to a given issue in a code repository.

{tool_descriptions}

## Exploration Strategy
Follow this general approach:
1. Start with one focused grep() using the most distinctive identifier from the issue.
2. Use list_dir() only when you need to understand a promising directory.
3. Use read_file() to inspect the most relevant candidate files, optionally with line ranges.
4. As soon as you identify the main source file, actively look for matching tests.
5. Stop early once you have enough evidence, then call finish().

## Response Format
On EVERY turn, respond in EXACTLY this format:
Thought: <your reasoning about what you know so far and what to do next>
Action: <tool_name>(arg1, arg2, ...)

When you have gathered sufficient evidence, respond:
Thought: <summary of the relevant files found and why>
Action: finish(file1.py, file2.py, ...)

## Rules
- Always start with a Thought.
- Call exactly ONE action per turn.
- If a tool returns an error or no results, try a different query or tool rather than repeating the same call.
- Never repeat the exact same tool call with the exact same arguments.
- Prefer 2-6 total tool calls; do not keep exploring once the likely source file and its test are known.
- The finish() arguments must be actual file paths only (e.g. "src/foo.py"). Do NOT pass descriptions, sentences, or list literals.
- The finish() arguments should include the file where the error likely is, and the test for that file if it exists.
- You have at most {max_steps} steps. If approaching the limit, call finish() with your best findings so far.
- You must always provide the test files for the relevant code, if they exist, as they often contain crucial information about how the code is used and what the expected behavior is.

/nothink
"""