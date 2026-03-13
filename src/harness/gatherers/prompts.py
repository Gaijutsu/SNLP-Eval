"""
Python file that gatherers can use to get their prompts
"""

# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------

AGENTLESS_FILE_LOCALIZATION_PROMPT = """\
You are an expert software engineer performing fault localization on a code repository.

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
Before writing the patch, briefly plan your fix:
- What is the root cause?
- What is the minimal change needed?
- Are there edge cases to handle?

Then generate the patch. Rules:
- Output a single unified diff block, starting with ```diff and ending with ```.
- Only change lines directly related to the fix; do not reformat unrelated code.
- Preserve the existing code style (indentation, naming conventions).
- If multiple files need changes, include all hunks in one diff block.

/nothink
"""


def get_agentless_file_localization_prompt():
    return AGENTLESS_FILE_LOCALIZATION_PROMPT


def get_agentless_function_localization_prompt():
    return AGENTLESS_FUNCTION_LOCALIZATION_PROMPT


def get_agentless_repair_prompt():
    return AGENTLESS_REPAIR_PROMPT


def get_agentless_prompts():
    return (
        get_agentless_file_localization_prompt,
        get_agentless_function_localization_prompt,
        get_agentless_repair_prompt,
    )


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

3. read_file(path: str) -> str
   Read the contents of a file (relative to repo root). Returns first 200 lines.
   Use this to confirm a file's relevance by inspecting its implementation.
   Example: read_file("src/auth/login.py")

/nothink
"""

REACT_SYSTEM_PROMPT = """\
You are a code investigation agent. Your goal is to identify the files most
relevant to a given issue in a code repository.

{tool_descriptions}

## Exploration Strategy
Follow this general approach:
1. Start with search_codebase() to get an initial set of candidate files.
2. Use list_dir() to understand the directory structure around candidates.
3. Use grep() to find where relevant symbols (functions, classes, errors) are defined or called.
4. Use read_file() to confirm relevance by inspecting implementation details.
5. Repeat until you have sufficient evidence, then call finish().

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
- The finish() arguments are the relevant file paths — only include files you have direct evidence for.
- Prefer precision: 3-10 high-confidence files is better than a long uncertain list.
- You have at most {max_steps} steps. If approaching the limit, call finish() with your best findings so far.

/nothink
"""


def get_react_tool_descriptions():
    return REACT_TOOL_DESCRIPTIONS


def get_react_system_prompt():
    return REACT_SYSTEM_PROMPT
