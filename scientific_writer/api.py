"""Async API for programmatic scientific document generation."""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator, Union, Literal
from datetime import datetime
from dotenv import load_dotenv

import requests

from claude_agent_sdk import query as claude_query, ClaudeAgentOptions
from claude_agent_sdk.types import HookMatcher, StopHookInput, HookContext

from .core import (
    resolve_provider_config,
    OPENAI_DEFAULT_BASE_URL,
    DEEPSEEK_DEFAULT_BASE_URL,
    load_system_instructions,
    ensure_output_folder,
    get_data_files,
    process_data_files,
    create_data_context_message,
    setup_claude_skills,
)
from .models import ProgressUpdate, TextUpdate, PaperResult, PaperMetadata, PaperFiles, TokenUsage
from .utils import (
    scan_paper_directory,
    count_citations_in_bib,
    extract_citation_style,
    count_words_in_tex,
    extract_title_from_tex,
)

# Model mapping for effort levels
EFFORT_LEVEL_MODELS = {
    "low": "claude-haiku-4-5",
    "medium": "claude-sonnet-4-5",
    "high": "claude-opus-4-5",
}

OPENAI_EFFORT_LEVEL_MODELS = {
    "low": "gpt-4o-mini",
    "medium": "gpt-4o",
    "high": "gpt-4.1",
}

DEEPSEEK_EFFORT_LEVEL_MODELS = {
    "low": "deepseek-chat",
    "medium": "deepseek-chat",
    "high": "deepseek-reasoner",
}


def create_completion_check_stop_hook(auto_continue: bool = True):
    """
    Create a stop hook that optionally forces continuation.
    
    Args:
        auto_continue: If True, always continue (never stop on agent's own).
                      If False, allow normal stopping behavior.
    """
    async def completion_check_stop_hook(
        hook_input: StopHookInput,
        matcher: str | None,
        context: HookContext,
    ) -> dict:
        """
        Stop hook that checks if the task is complete before allowing stop.
        
        When auto_continue is True, this returns continue_=True to force
        the agent to continue working instead of stopping.
        """
        if auto_continue:
            # Force continuation - the agent should not stop on its own
            return {"continue_": True}
        
        # Allow the stop
        return {"continue_": False}
    
    return completion_check_stop_hook


async def generate_paper(
    query: str,
    output_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    effort_level: Literal["low", "medium", "high"] = "medium",
    data_files: Optional[List[str]] = None,
    cwd: Optional[str] = None,
    track_token_usage: bool = False,
    auto_continue: bool = True,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Generate a scientific document asynchronously with progress updates.
    
    This is a stateless async generator that yields progress updates during
    execution and a final comprehensive result with all document details.
    Supports papers, slides, posters, reports, grants, and other document types.
    
    Args:
        query: The document generation request (e.g., "Create a Nature paper on CRISPR",
               "Generate conference slides on AI", "Create a research poster")
        output_dir: Optional custom output directory (defaults to cwd/writing_outputs)
        api_key: Optional API key (defaults to SCIENTIFIC_WRITER_API_KEY/ANTHROPIC_API_KEY env var)
        provider: Optional provider name (defaults to SCIENTIFIC_WRITER_PROVIDER, or "anthropic")
        base_url: Optional base URL (defaults to SCIENTIFIC_WRITER_BASE_URL)
        model: Optional explicit Claude model to use. If provided, overrides effort_level.
        effort_level: Effort level that determines the model to use (default: "medium"):
            - "low": Uses Claude Haiku 4.5 (fastest, most economical)
            - "medium": Uses Claude Sonnet 4.5 (balanced) [default]
            - "high": Uses Claude Opus 4.5 (most capable)
        data_files: Optional list of data file paths to include
        cwd: Optional working directory (defaults to package parent directory)
        track_token_usage: If True, track and return token usage in the final result
        auto_continue: If True (default), the agent will not stop on its own and will
            continue working until the task is complete. Set to False to allow
            normal stopping behavior.
    
    Yields:
        Progress updates (dict with type="progress") during execution
        Final result (dict with type="result") containing all document information
        
    Example:
        ```python
        async for update in generate_paper("Create a NeurIPS paper on transformers"):
            if update["type"] == "progress":
                print(f"[{update['stage']}] {update['message']}")
            else:
                print(f"Document created: {update['paper_directory']}")
                print(f"PDF: {update['files']['pdf_final']}")
        
        # With token usage tracking:
        async for update in generate_paper("Create a paper", track_token_usage=True):
            if update["type"] == "result":
                print(f"Token usage: {update.get('token_usage')}")
        ```
    """
    # Initialize
    start_time = time.time()
    
    # Explicitly load .env file from working directory
    # Determine working directory first
    if cwd:
        work_dir = Path(cwd).resolve()
    else:
        work_dir = Path.cwd().resolve()
    
    # Load .env from working directory
    env_file = work_dir / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=True)
    
    # Resolve provider configuration (also sets provider-specific env vars for SDKs)
    try:
        provider_config = resolve_provider_config(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
        )
    except ValueError as e:
        yield _create_error_result(str(e))
        return
    
    # Resolve model: explicit model parameter takes precedence, otherwise use effort_level
    if model is None:
        if provider_config.provider == "openai":
            model = OPENAI_EFFORT_LEVEL_MODELS[effort_level]
        elif provider_config.provider == "deepseek":
            model = DEEPSEEK_EFFORT_LEVEL_MODELS[effort_level]
        else:
            model = EFFORT_LEVEL_MODELS[effort_level]
    
    # Get package directory for copying skills to working directory
    package_dir = Path(__file__).parent.absolute()  # scientific_writer/ directory
    
    # Set up Claude skills in the working directory (includes WRITER.md)
    setup_claude_skills(package_dir, work_dir)
    
    # Ensure output folder exists in user's directory
    output_folder = ensure_output_folder(work_dir, output_dir)
    
    # Initial progress update
    yield ProgressUpdate(
        message="Initializing document generation",
        stage="initialization",
    ).to_dict()
    
    # Load system instructions from .claude/WRITER.md in working directory
    system_instructions = load_system_instructions(work_dir)
    
    # Add conversation continuity instruction
    system_instructions += "\n\n" + f"""
IMPORTANT - WORKING DIRECTORY:
- Your working directory is: {work_dir}
- ALWAYS create writing_outputs folder in this directory: {work_dir}/writing_outputs/
- NEVER write to /tmp/ or any other temporary directory
- All paper outputs MUST go to: {work_dir}/writing_outputs/<timestamp>_<description>/

IMPORTANT - CONVERSATION CONTINUITY:
- This is a NEW paper request - create a new paper directory
- Create a unique timestamped directory in the writing_outputs folder
- Do NOT assume there's an existing paper unless explicitly told in the prompt context
"""
    
    # Process data files if provided
    data_context = ""
    temp_paper_path = None
    
    if data_files:
        data_file_paths = get_data_files(work_dir, data_files)
        if data_file_paths:
            # We'll need to process these after the output directory is created
            yield ProgressUpdate(
                message=f"Found {len(data_file_paths)} data file(s) to process",
                stage="initialization",
            ).to_dict()
    
    # Check if auto-continue is enabled (parameter takes precedence over env var)
    # Environment variable can override if parameter is True (default)
    env_auto_continue = os.environ.get("SCIENTIFIC_WRITER_AUTO_CONTINUE", "").lower()
    if env_auto_continue in ("false", "0", "no"):
        auto_continue = False
    
    # OpenAI-compatible providers use a local tool runner instead of Claude agent SDK
    if provider_config.provider in ("openai", "deepseek"):
        async for update in _run_openai_compatible_agent(
            query=query,
            system_instructions=system_instructions,
            model=model,
            provider_config=provider_config,
            work_dir=work_dir,
            output_folder=output_folder,
            package_dir=package_dir,
            start_time=start_time,
            data_files=data_files,
            track_token_usage=track_token_usage,
            auto_continue=auto_continue,
        ):
            yield update
        return
    
    # Configure Claude agent options with stop hook for completion checking
    options = ClaudeAgentOptions(
        system_prompt=system_instructions,
        model=model,
        allowed_tools=["Read", "Write", "Edit", "Bash", "WebSearch", "research-lookup"],
        permission_mode="bypassPermissions",
        setting_sources=["project"],  # Load skills from project .claude directory
        cwd=str(work_dir),  # User's working directory
        max_turns=500,  # Allow many turns for long document generation
        hooks={
            "Stop": [
                HookMatcher(
                    matcher=None,  # Match all stop events
                    hooks=[create_completion_check_stop_hook(auto_continue=auto_continue)],
                )
            ]
        },
    )
    
    # Track progress through message analysis
    current_stage = "initialization"
    output_directory = None
    last_message = ""  # Track last message to avoid duplicates
    tool_call_count = 0
    files_written = []
    
    # Token usage tracking (when enabled)
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_creation_tokens = 0
    total_cache_read_tokens = 0
    
    yield ProgressUpdate(
        message="Starting document generation",
        stage="initialization",
        details={"query_length": len(query)},
    ).to_dict()
    
    # Execute query
    try:
        accumulated_text = ""
        async for message in claude_query(prompt=query, options=options):
            # Track token usage if enabled
            if track_token_usage and hasattr(message, "usage") and message.usage:
                usage = message.usage
                total_input_tokens += getattr(usage, "input_tokens", 0)
                total_output_tokens += getattr(usage, "output_tokens", 0)
                total_cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0)
                total_cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0)
            
            if hasattr(message, "content") and message.content:
                for block in message.content:
                    # Handle text blocks - stream live and analyze for progress
                    if hasattr(block, "text"):
                        text = block.text
                        accumulated_text += text
                        
                        # Yield live text update - stream Scientific-Writer's actual response
                        yield TextUpdate(content=text).to_dict()
                        
                        # Analyze text for major stage transitions (fallback)
                        stage, msg = _analyze_progress(accumulated_text, current_stage)
                        
                        # Only yield progress if we have a stage change with a message
                        if stage != current_stage and msg and msg != last_message:
                            current_stage = stage
                            last_message = msg
                            
                            yield ProgressUpdate(
                                message=msg,
                                stage=stage,
                            ).to_dict()
                    
                    # Handle tool use blocks - provide detailed progress on actions
                    elif hasattr(block, "type") and block.type == "tool_use":
                        tool_call_count += 1
                        tool_name = getattr(block, "name", "unknown")
                        tool_input = getattr(block, "input", {})
                        
                        # Track files being written
                        if tool_name.lower() == "write":
                            file_path = tool_input.get("file_path", tool_input.get("path", ""))
                            if file_path:
                                files_written.append(file_path)
                        
                        # Analyze tool usage for progress
                        tool_progress = _analyze_tool_use(tool_name, tool_input, current_stage)
                        
                        if tool_progress:
                            stage, msg = tool_progress
                            if msg != last_message:
                                current_stage = stage
                                last_message = msg
                                
                                yield ProgressUpdate(
                                    message=msg,
                                    stage=stage,
                                    details={
                                        "tool": tool_name,
                                        "tool_calls": tool_call_count,
                                        "files_created": len(files_written),
                                    },
                                ).to_dict()
        
        # Document generation complete - now scan for results
        yield ProgressUpdate(
            message="Scanning output directory",
            stage="complete",
        ).to_dict()
        
        # Find the most recently created output directory
        output_directory = _find_most_recent_output(output_folder, start_time)
        
        if not output_directory:
            error_result = _create_error_result("Output directory not found after generation")
            if track_token_usage:
                error_result['token_usage'] = TokenUsage(
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    cache_creation_input_tokens=total_cache_creation_tokens,
                    cache_read_input_tokens=total_cache_read_tokens,
                ).to_dict()
            yield error_result
            return
        
        # Process any data files now if we have an output directory
        if data_files:
            data_file_paths = get_data_files(work_dir, data_files)
            if data_file_paths:
                processed_info = process_data_files(
                    work_dir, 
                    data_file_paths, 
                    str(output_directory),
                    delete_originals=False  # Don't delete when using programmatic API
                )
                if processed_info:
                    manuscript_count = len(processed_info.get('manuscript_files', []))
                    message = f"Processed {len(processed_info['all_files'])} file(s)"
                    if manuscript_count > 0:
                        message += f" ({manuscript_count} manuscript(s) copied to drafts/)"
                    yield ProgressUpdate(
                        message=message,
                        stage="complete",
                    ).to_dict()
        
        # Scan the output directory for all files
        file_info = scan_paper_directory(output_directory)
        
        # Build comprehensive result
        result = _build_paper_result(output_directory, file_info)
        
        # Add token usage if tracking is enabled
        if track_token_usage:
            result.token_usage = TokenUsage(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cache_creation_input_tokens=total_cache_creation_tokens,
                cache_read_input_tokens=total_cache_read_tokens,
            )
        
        yield ProgressUpdate(
            message="Document generation complete",
            stage="complete",
        ).to_dict()
        
        # Final result
        yield result.to_dict()
        
    except Exception as e:
        error_result = _create_error_result(f"Error during document generation: {str(e)}")
        # Include token usage even on error if tracking was enabled
        if track_token_usage:
            error_result['token_usage'] = TokenUsage(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cache_creation_input_tokens=total_cache_creation_tokens,
                cache_read_input_tokens=total_cache_read_tokens,
            ).to_dict()
        yield error_result


def _analyze_progress(text: str, current_stage: str) -> tuple:
    """
    Minimal fallback for progress detection from text.
    
    Primary progress updates come from tool usage analysis (_analyze_tool_use).
    This function only detects major stage transitions when no tool updates available.
    
    Returns:
        Tuple of (stage, message) - returns current stage if no transition detected
    """
    text_lower = text.lower()
    
    # Stage order for progression tracking
    stage_order = ["initialization", "planning", "research", "writing", "compilation", "complete"]
    current_idx = stage_order.index(current_stage) if current_stage in stage_order else 0
    
    # Only detect major stage transitions - let tool analysis handle specifics
    # Check for compilation indicators (most definitive)
    if current_idx < stage_order.index("compilation"):
        if "pdflatex" in text_lower or "latexmk" in text_lower or "compiling" in text_lower:
            return "compilation", "Compiling document"
    
    # Check for completion indicators
    if current_idx < stage_order.index("complete"):
        if "successfully compiled" in text_lower or "pdf generated" in text_lower:
            return "complete", "Finalizing output"
    
    # No stage transition detected - return current stage without message change
    return current_stage, None


def _detect_document_type(file_path: str) -> str:
    """Detect document type from file path."""
    path_lower = file_path.lower()
    if "slide" in path_lower or "presentation" in path_lower or "beamer" in path_lower:
        return "slides"
    elif "poster" in path_lower:
        return "poster"
    elif "report" in path_lower:
        return "report"
    elif "grant" in path_lower or "proposal" in path_lower:
        return "grant"
    return "document"


def _get_section_from_filename(filename: str) -> str:
    """Extract section name from filename for more descriptive messages."""
    name_lower = filename.lower().replace('.tex', '').replace('.md', '')
    
    section_mappings = {
        'abstract': 'abstract',
        'intro': 'introduction',
        'introduction': 'introduction',
        'method': 'methods',
        'methods': 'methods',
        'methodology': 'methodology',
        'result': 'results',
        'results': 'results',
        'discussion': 'discussion',
        'conclusion': 'conclusion',
        'conclusions': 'conclusions',
        'background': 'background',
        'related': 'related work',
        'experiment': 'experiments',
        'experiments': 'experiments',
        'evaluation': 'evaluation',
        'appendix': 'appendix',
        'supplement': 'supplementary material',
    }
    
    for key, section in section_mappings.items():
        if key in name_lower:
            return section
    return None


def _analyze_tool_use(tool_name: str, tool_input: Dict[str, Any], current_stage: str) -> tuple:
    """
    Analyze tool usage to provide dynamic, context-aware progress updates.
    
    Args:
        tool_name: Name of the tool being used
        tool_input: Input parameters to the tool
        current_stage: Current progress stage
        
    Returns:
        Tuple of (stage, message) or None if no update needed
    """
    # Stage order for progression
    stage_order = ["initialization", "planning", "research", "writing", "compilation", "complete"]
    current_idx = stage_order.index(current_stage) if current_stage in stage_order else 0
    
    # Extract relevant info from tool input
    file_path = tool_input.get("file_path", tool_input.get("path", ""))
    command = tool_input.get("command", "")
    filename = Path(file_path).name if file_path else ""
    doc_type = _detect_document_type(file_path)
    
    # Read tool - detect what's being read
    if tool_name.lower() == "read":
        if ".bib" in file_path:
            return ("writing", f"Reading bibliography: {filename}")
        elif ".tex" in file_path:
            section = _get_section_from_filename(filename)
            if section:
                return ("writing", f"Reading {section} section")
            return ("writing", f"Reading {filename}")
        elif ".pdf" in file_path:
            return ("research", f"Analyzing PDF: {filename}")
        elif ".csv" in file_path:
            return ("research", f"Loading data from {filename}")
        elif ".json" in file_path:
            return ("research", f"Reading configuration: {filename}")
        elif ".md" in file_path:
            return ("planning", f"Reading {filename}")
        elif file_path:
            return (current_stage, f"Reading {filename}")
        return None
    
    # Write tool - detect what's being written
    elif tool_name.lower() == "write":
        if ".bib" in file_path:
            return ("writing", f"Creating bibliography with references")
        elif ".tex" in file_path:
            section = _get_section_from_filename(filename)
            if section:
                return ("writing", f"Writing {section} section")
            elif "main" in filename.lower():
                return ("writing", f"Creating main {doc_type} structure")
            elif current_idx < stage_order.index("writing"):
                return ("writing", f"Writing {doc_type}: {filename}")
            else:
                return ("compilation", f"Updating {filename}")
        elif ".md" in file_path:
            if "progress" in filename.lower():
                return ("writing", "Updating progress log")
            elif "readme" in filename.lower():
                return ("complete", "Creating documentation")
            return ("writing", f"Writing {filename}")
        elif ".sty" in file_path:
            return ("writing", f"Creating style file: {filename}")
        elif ".cls" in file_path:
            return ("writing", f"Creating document class: {filename}")
        elif file_path:
            return (current_stage, f"Creating {filename}")
        return None
    
    # Edit tool
    elif tool_name.lower() == "edit":
        if ".tex" in file_path:
            section = _get_section_from_filename(filename)
            if section:
                return ("writing", f"Refining {section} section")
            return ("writing", f"Editing {filename}")
        elif ".bib" in file_path:
            return ("writing", "Updating bibliography")
        elif file_path:
            return (current_stage, f"Editing {filename}")
        return None
    
    # Bash tool - detect compilation and other commands
    elif tool_name.lower() == "bash":
        if "pdflatex" in command:
            # Try to extract filename from command
            if "-output-directory" in command:
                return ("compilation", "Compiling PDF with output directory")
            return ("compilation", "Compiling LaTeX to PDF")
        elif "latexmk" in command:
            return ("compilation", "Running full LaTeX compilation pipeline")
        elif "bibtex" in command:
            return ("compilation", "Processing bibliography citations")
        elif "makeindex" in command:
            return ("compilation", "Building document index")
        elif "mkdir" in command:
            # Try to extract directory purpose
            if "writing_outputs" in command or "output" in command.lower():
                return ("initialization", "Creating output directory")
            elif "figures" in command.lower():
                return ("initialization", "Setting up figures directory")
            elif "drafts" in command.lower():
                return ("initialization", "Setting up drafts directory")
            return ("initialization", "Creating directory structure")
        elif "cp " in command:
            if ".pdf" in command:
                return ("complete", "Copying final PDF to output")
            elif ".tex" in command:
                return ("complete", "Archiving LaTeX source")
            return ("complete", "Organizing files")
        elif "mv " in command:
            return ("complete", "Moving files to final location")
        elif "ls " in command or "cat " in command:
            return None  # Don't report on inspection commands
        elif command:
            # Truncate long commands intelligently
            cmd_preview = command.split()[0] if command.split() else command[:30]
            return (current_stage, f"Running {cmd_preview}")
        return None
    
    # Research lookup tool
    elif "research" in tool_name.lower() or "lookup" in tool_name.lower():
        query_text = tool_input.get("query", "")
        if query_text:
            # Truncate but keep meaningful content
            truncated = query_text[:50] + "..." if len(query_text) > 50 else query_text
            return ("research", f"Searching: {truncated}")
        return ("research", "Searching literature databases")
    
    # Web search or similar tools
    elif "search" in tool_name.lower() or "web" in tool_name.lower():
        query_text = tool_input.get("query", tool_input.get("search_term", ""))
        if query_text:
            truncated = query_text[:40] + "..." if len(query_text) > 40 else query_text
            return ("research", f"Web search: {truncated}")
        return ("research", "Searching online resources")
    
    return None


def _normalize_openai_base_url(provider: str, base_url: Optional[str]) -> str:
    if base_url:
        resolved = base_url.rstrip("/")
    else:
        if provider == "deepseek":
            resolved = DEEPSEEK_DEFAULT_BASE_URL
        else:
            resolved = OPENAI_DEFAULT_BASE_URL
        resolved = resolved.rstrip("/")
    if not resolved.endswith("/v1"):
        resolved = f"{resolved}/v1"
    return resolved


def _openai_tools_schema() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read a UTF-8 text file from disk (paths are relative to the working directory).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Write",
                "description": "Write text content to a file (creates directories as needed).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to write"},
                        "content": {"type": "string", "description": "Full file contents"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Edit",
                "description": "Edit a text file by replacing old_text with new_text (or applying multiple edits).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to edit"},
                        "old_text": {"type": "string", "description": "Text to replace"},
                        "new_text": {"type": "string", "description": "Replacement text"},
                        "edits": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "old_text": {"type": "string"},
                                    "new_text": {"type": "string"},
                                },
                                "required": ["old_text", "new_text"],
                            },
                        },
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "Run a shell command in the working directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to run"},
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "research-lookup",
                "description": "Run the research-lookup tool for scholarly search (requires OPENROUTER_API_KEY).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Research query"},
                    },
                    "required": ["query"],
                },
            },
        },
    ]


def _augment_system_instructions_for_openai(system_instructions: str, work_dir: Path) -> str:
    return (
        system_instructions
        + "\n\nOPENAI TOOLING (LOCAL EXECUTION):\n"
        + f"- Working directory: {work_dir}\n"
        + "- Use these tools exactly: Read, Write, Edit, Bash, research-lookup.\n"
        + "- Paths are relative to the working directory. Do NOT write outside it.\n"
        + "- Edit supports either {old_text, new_text} or edits=[{old_text,new_text}, ...].\n"
        + "- research-lookup uses OpenRouter and requires OPENROUTER_API_KEY.\n"
    )


def _openai_chat_completion_request(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    timeout: int = 180,
) -> Dict[str, Any]:
    url = f"{base_url}/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


async def _openai_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return await asyncio.to_thread(
        _openai_chat_completion_request,
        base_url,
        api_key,
        model,
        messages,
        tools,
    )


def _truncate_tool_output(text: str, limit: int = 6000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def _resolve_tool_path(work_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = work_dir / path
    return path.resolve()


def _execute_openai_tool(
    name: str,
    args: Dict[str, Any],
    work_dir: Path,
    package_dir: Path,
) -> str:
    tool_name = name.lower()
    if tool_name == "read":
        raw_path = args.get("path") or args.get("file_path")
        if not raw_path:
            return "Error: missing path"
        resolved = _resolve_tool_path(work_dir, raw_path)
        if work_dir not in resolved.parents and resolved != work_dir:
            return f"Error: path outside working directory: {resolved}"
        if not resolved.exists():
            return f"Error: file not found: {resolved}"
        try:
            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return _truncate_tool_output(content)
        except Exception as e:
            return f"Error reading file: {e}"
    
    if tool_name == "write":
        raw_path = args.get("path") or args.get("file_path")
        content = args.get("content", "")
        if not raw_path:
            return "Error: missing path"
        resolved = _resolve_tool_path(work_dir, raw_path)
        if work_dir not in resolved.parents and resolved != work_dir:
            return f"Error: path outside working directory: {resolved}"
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(content)
            return f"OK: wrote {resolved}"
        except Exception as e:
            return f"Error writing file: {e}"
    
    if tool_name == "edit":
        raw_path = args.get("path") or args.get("file_path")
        if not raw_path:
            return "Error: missing path"
        resolved = _resolve_tool_path(work_dir, raw_path)
        if work_dir not in resolved.parents and resolved != work_dir:
            return f"Error: path outside working directory: {resolved}"
        if not resolved.exists():
            return f"Error: file not found: {resolved}"
        try:
            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                original = f.read()
            edits = args.get("edits")
            if edits:
                updated = original
                for edit in edits:
                    old_text = edit.get("old_text", "")
                    new_text = edit.get("new_text", "")
                    if old_text not in updated:
                        return "Error: old_text not found for one edit"
                    updated = updated.replace(old_text, new_text, 1)
            else:
                old_text = args.get("old_text", "")
                new_text = args.get("new_text", "")
                if not old_text:
                    return "Error: missing old_text"
                if old_text not in original:
                    return "Error: old_text not found"
                updated = original.replace(old_text, new_text, 1)
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(updated)
            return f"OK: edited {resolved}"
        except Exception as e:
            return f"Error editing file: {e}"
    
    if tool_name == "bash":
        command = args.get("command", "")
        if not command:
            return "Error: missing command"
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                check=False,
            )
            output = result.stdout + result.stderr
            summary = f"exit_code={result.returncode}\n{output}"
            return _truncate_tool_output(summary)
        except Exception as e:
            return f"Error running command: {e}"
    
    if tool_name in ("research-lookup", "research_lookup", "research"):
        query = args.get("query", "")
        if not query:
            return "Error: missing query"
        lookup_script = work_dir / ".claude" / "skills" / "research-lookup" / "lookup.py"
        if not lookup_script.exists():
            lookup_script = package_dir / ".claude" / "skills" / "research-lookup" / "lookup.py"
        if not lookup_script.exists():
            return "Error: research-lookup tool not found"
        try:
            result = subprocess.run(
                [sys.executable, str(lookup_script), query],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                check=False,
            )
            output = result.stdout + result.stderr
            return _truncate_tool_output(output)
        except Exception as e:
            return f"Error running research-lookup: {e}"
    
    return f"Error: unknown tool '{name}'"


async def _run_openai_compatible_agent(
    query: str,
    system_instructions: str,
    model: str,
    provider_config: Any,
    work_dir: Path,
    output_folder: Path,
    package_dir: Path,
    start_time: float,
    data_files: Optional[List[str]],
    track_token_usage: bool,
    auto_continue: bool,
) -> AsyncGenerator[Dict[str, Any], None]:
    current_stage = "initialization"
    last_message = ""
    tool_call_count = 0
    files_written: List[str] = []
    accumulated_text = ""
    
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_creation_tokens = 0
    total_cache_read_tokens = 0
    
    yield ProgressUpdate(
        message="Starting document generation",
        stage="initialization",
        details={"query_length": len(query), "provider": provider_config.provider},
    ).to_dict()
    
    tools = _openai_tools_schema()
    system_prompt = _augment_system_instructions_for_openai(system_instructions, work_dir)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    
    base_url = _normalize_openai_base_url(provider_config.provider, provider_config.base_url)
    
    try:
        for _ in range(500):
            response = await _openai_chat_completion(
                base_url=base_url,
                api_key=provider_config.api_key,
                model=model,
                messages=messages,
                tools=tools,
            )
            
            usage = response.get("usage") or {}
            if track_token_usage:
                total_input_tokens += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
                total_output_tokens += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
            
            choice = response.get("choices", [{}])[0].get("message", {})
            content = choice.get("content") or ""
            tool_calls = choice.get("tool_calls") or []
            
            assistant_message: Dict[str, Any] = {"role": "assistant"}
            if content:
                accumulated_text += content
                assistant_message["content"] = content
                yield TextUpdate(content=content).to_dict()
                
                stage, msg = _analyze_progress(accumulated_text, current_stage)
                if stage != current_stage and msg and msg != last_message:
                    current_stage = stage
                    last_message = msg
                    yield ProgressUpdate(message=msg, stage=stage).to_dict()
            
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            messages.append(assistant_message)
            
            if not tool_calls:
                break
            
            for tool_call in tool_calls:
                tool_call_count += 1
                function = tool_call.get("function", {})
                name = function.get("name", "")
                raw_args = function.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except json.JSONDecodeError:
                    args = {"_raw": raw_args}
                
                # Track files being written
                if name.lower() == "write":
                    file_path = args.get("path") or args.get("file_path", "")
                    if file_path:
                        files_written.append(file_path)
                
                tool_progress = _analyze_tool_use(name, args, current_stage)
                if tool_progress:
                    stage, msg = tool_progress
                    if msg != last_message:
                        current_stage = stage
                        last_message = msg
                        yield ProgressUpdate(
                            message=msg,
                            stage=stage,
                            details={
                                "tool": name,
                                "tool_calls": tool_call_count,
                                "files_created": len(files_written),
                            },
                        ).to_dict()
                
                result = _execute_openai_tool(name, args, work_dir, package_dir)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "content": result,
                    }
                )
        
        yield ProgressUpdate(message="Scanning output directory", stage="complete").to_dict()
        output_directory = _find_most_recent_output(output_folder, start_time)
        
        if not output_directory:
            error_result = _create_error_result("Output directory not found after generation")
            if track_token_usage:
                error_result["token_usage"] = TokenUsage(
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    cache_creation_input_tokens=total_cache_creation_tokens,
                    cache_read_input_tokens=total_cache_read_tokens,
                ).to_dict()
            yield error_result
            return
        
        if data_files:
            data_file_paths = get_data_files(work_dir, data_files)
            if data_file_paths:
                processed_info = process_data_files(
                    work_dir,
                    data_file_paths,
                    str(output_directory),
                    delete_originals=False,
                )
                if processed_info:
                    manuscript_count = len(processed_info.get("manuscript_files", []))
                    message = f"Processed {len(processed_info['all_files'])} file(s)"
                    if manuscript_count > 0:
                        message += f" ({manuscript_count} manuscript(s) copied to drafts/)"
                    yield ProgressUpdate(message=message, stage="complete").to_dict()
        
        file_info = scan_paper_directory(output_directory)
        result = _build_paper_result(output_directory, file_info)
        
        if track_token_usage:
            result.token_usage = TokenUsage(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cache_creation_input_tokens=total_cache_creation_tokens,
                cache_read_input_tokens=total_cache_read_tokens,
            )
        
        yield ProgressUpdate(message="Document generation complete", stage="complete").to_dict()
        yield result.to_dict()
    
    except Exception as e:
        error_result = _create_error_result(f"Error during document generation: {str(e)}")
        if track_token_usage:
            error_result["token_usage"] = TokenUsage(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cache_creation_input_tokens=total_cache_creation_tokens,
                cache_read_input_tokens=total_cache_read_tokens,
            ).to_dict()
        yield error_result


def _find_most_recent_output(output_folder: Path, start_time: float) -> Optional[Path]:
    """
    Find the most recently created/modified output directory.
    
    Args:
        output_folder: Path to output folder
        start_time: Start time of generation (to filter relevant directories)
    
    Returns:
        Path to output directory or None
    """
    try:
        output_dirs = [d for d in output_folder.iterdir() if d.is_dir()]
        if not output_dirs:
            return None
        
        # Filter to only directories modified after start_time
        recent_dirs = [
            d for d in output_dirs 
            if d.stat().st_mtime >= start_time - 5  # 5 second buffer
        ]
        
        if not recent_dirs:
            # Fallback to most recent directory overall
            recent_dirs = output_dirs
        
        # Return the most recent
        most_recent = max(recent_dirs, key=lambda d: d.stat().st_mtime)
        return most_recent
    except Exception:
        return None


def _build_paper_result(paper_dir: Path, file_info: Dict[str, Any]) -> PaperResult:
    """
    Build a comprehensive PaperResult from scanned files.
    
    Args:
        paper_dir: Path to paper directory
        file_info: Dictionary of file information from scan_paper_directory
    
    Returns:
        PaperResult object
    """
    # Extract metadata
    tex_file = file_info['tex_final'] or (file_info['tex_drafts'][0] if file_info['tex_drafts'] else None)
    
    title = extract_title_from_tex(tex_file)
    word_count = count_words_in_tex(tex_file)
    
    # Extract topic from directory name
    topic = ""
    parts = paper_dir.name.split('_', 2)
    if len(parts) >= 3:
        topic = parts[2].replace('_', ' ')
    
    metadata = PaperMetadata(
        title=title,
        created_at=datetime.fromtimestamp(paper_dir.stat().st_ctime).isoformat() + "Z",
        topic=topic,
        word_count=word_count,
    )
    
    # Build files object
    files = PaperFiles(
        pdf_final=file_info['pdf_final'],
        tex_final=file_info['tex_final'],
        pdf_drafts=file_info['pdf_drafts'],
        tex_drafts=file_info['tex_drafts'],
        bibliography=file_info['bibliography'],
        figures=file_info['figures'],
        data=file_info['data'],
        progress_log=file_info['progress_log'],
        summary=file_info['summary'],
    )
    
    # Citations info
    citation_count = count_citations_in_bib(file_info['bibliography'])
    citation_style = extract_citation_style(file_info['bibliography'])
    
    citations = {
        'count': citation_count,
        'style': citation_style,
        'file': file_info['bibliography'],
    }
    
    # Determine status
    status = "success"
    compilation_success = file_info['pdf_final'] is not None
    
    if not compilation_success:
        if file_info['tex_final']:
            status = "partial"  # TeX created but PDF failed
        else:
            status = "failed"
    
    result = PaperResult(
        status=status,
        paper_directory=str(paper_dir),
        paper_name=paper_dir.name,
        metadata=metadata,
        files=files,
        citations=citations,
        figures_count=len(file_info['figures']),
        compilation_success=compilation_success,
        errors=[],
    )
    
    return result


def _create_error_result(error_message: str) -> Dict[str, Any]:
    """
    Create an error result dictionary.
    
    Args:
        error_message: Error message string
    
    Returns:
        Dictionary with error information
    """
    result = PaperResult(
        status="failed",
        paper_directory="",
        paper_name="",
        errors=[error_message],
    )
    return result.to_dict()
