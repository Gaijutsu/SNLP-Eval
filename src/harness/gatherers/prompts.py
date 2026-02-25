"""
Python file thjat gathjerers can use to get their prompts
"""

# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------

AGENTLESS_FILE_LOCALIZATION_PROMPT = """\
You are an expert software engineer. Given the following issue/bug report
and repository file listing, identify which files are most likely relevant
to this issue. Return ONLY a JSON array of file paths, ranked by relevance.

## Issue
{query}

## Repository Files
{file_listing}

Respond with a JSON array of up to {top_n} file paths, most relevant first.
Example: ["src/auth/login.py", "src/models/user.py"]
"""

AGENTLESS_FUNCTION_LOCALIZATION_PROMPT = """\
You are an expert software engineer. Given the following issue and file
contents, identify the specific functions/classes/code regions that need
to be modified to fix this issue. Return a JSON array of objects with
"file" and "region" keys.

## Issue
{query}

## File Contents
{file_contents}

Respond with a JSON array of objects, each having:
- "file": the file path
- "region": description of the specific function/class/code region

Example: [{{"file": "src/auth.py", "region": "def login() around line 42"}}]
"""

AGENTLESS_REPAIR_PROMPT = """\
You are an expert software engineer. Given the issue description and the
relevant code regions, generate a patch in unified diff format to fix
the issue.

## Issue
{query}

## Relevant Code
{code_regions}

Generate a minimal, correct patch in unified diff format.
Start your response with ```diff and end with ```.
"""


def get_agentless_file_localization_prompt():
    return AGENTLESS_FILE_LOCALIZATION_PROMPT


def get_agentless_function_locatization_prompt():
    return AGENTLESS_FUNCTION_LOCALIZATION_PROMPT


def get_agentless_repair_prompt():
    return AGENTLESS_REPAIR_PROMPT


def get_agentless_prompts():
    return (
        get_agentless_file_localization_prompt,
        get_agentless_function_locatization_prompt,
        get_agentless_repair_prompt,
    )


REACT_TOOL_DESCRIPTIONS = """\
You have the following tools available:

1. list_dir(path: str) -> str
   List files and subdirectories in the given directory (relative to repo root).

2. read_file(path: str) -> str
   Read the contents of a file (relative to repo root). Returns first 200 lines.

3. grep(pattern: str, path: str = ".") -> str
   Search for a regex pattern in files under the given path. Returns matching lines.

4. search_codebase(query: str) -> str
   Semantic search over all files in the repo. Returns the top-5 most relevant file paths.
"""

REACT_SYSTEM_PROMPT = """\
You are a code investigation agent. Your job is to find the files most relevant
to a given issue/query in a code repository.

{tool_descriptions}

On each turn, respond in EXACTLY this format:
Thought: <your reasoning about what to do next>
Action: <tool_name>(arg1, arg2, ...)

When you have found all relevant files, respond:
Thought: <summary of findings>
Action: finish(file1.py, file2.py, ...)

Rules:
- Always start with a Thought.
- Call exactly ONE action per turn.
- The finish action's arguments are the relevant file paths you found.
- You have at most {max_steps} steps.
"""


def get_react_tool_descriptions():
    return REACT_TOOL_DESCRIPTIONS


def get_react_system_prompt():
    return REACT_SYSTEM_PROMPT
