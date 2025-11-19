"""
Context Manager - Intelligent Context Extraction and Token Optimization
Provides AST-based code analysis and context compression for efficient token usage
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class FunctionSignature:
    """Represents a function signature extracted from code"""
    name: str
    args: List[str]
    returns: Optional[str]
    docstring: Optional[str]
    line_number: int
    is_async: bool = False


@dataclass
class ClassSignature:
    """Represents a class signature extracted from code"""
    name: str
    bases: List[str]
    methods: List[FunctionSignature]
    docstring: Optional[str]
    line_number: int


@dataclass
class CodebaseContext:
    """Structured context extracted from codebase"""
    classes: List[ClassSignature]
    functions: List[FunctionSignature]
    imports: List[str]
    file_path: str
    total_lines: int


class ContextManager:
    """
    Manages context extraction and compression for token efficiency.

    This class extracts only relevant information from codebases:
    - Function signatures (no bodies)
    - Class structures (no method bodies)
    - Docstrings
    - Import statements

    This dramatically reduces token usage while maintaining semantic understanding.
    """

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {'.py'}

    # Maximum tokens for compact history (approximate)
    DEFAULT_MAX_TOKENS = 4000

    def __init__(self):
        """Initialize the Context Manager"""
        self.cached_contexts: Dict[str, CodebaseContext] = {}
        logger.info("ContextManager initialized")

    def extract_relevant_context(
        self,
        codebase_path: str,
        task_description: Optional[str] = None,
        include_bodies: bool = False
    ) -> str:
        """
        Extract relevant context from a codebase path.

        Args:
            codebase_path: Path to codebase (file or directory)
            task_description: Optional task description to filter context
            include_bodies: If True, include full function/method bodies

        Returns:
            Formatted context string optimized for LLM consumption
        """
        path = Path(codebase_path)

        if not path.exists():
            logger.error(f"Path does not exist: {codebase_path}")
            return f"# Error: Path not found: {codebase_path}\n"

        # Extract context from all Python files
        contexts = []

        if path.is_file() and path.suffix in self.SUPPORTED_EXTENSIONS:
            context = self._parse_python_file(str(path))
            if context:
                contexts.append(context)
        elif path.is_dir():
            for py_file in path.rglob("*.py"):
                # Skip __pycache__ and other ignored directories
                if "__pycache__" in str(py_file) or ".git" in str(py_file):
                    continue

                context = self._parse_python_file(str(py_file))
                if context:
                    contexts.append(context)

        # Format contexts into readable string
        formatted = self._format_contexts(contexts, include_bodies)

        logger.info(
            f"Extracted context from {len(contexts)} files "
            f"({len(formatted)} chars)"
        )

        return formatted

    def _parse_python_file(self, file_path: str) -> Optional[CodebaseContext]:
        """
        Parse a Python file using AST to extract signatures.

        Args:
            file_path: Path to Python file

        Returns:
            CodebaseContext or None if parsing failed
        """
        # Check cache first
        if file_path in self.cached_contexts:
            return self.cached_contexts[file_path]

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            # Parse AST
            tree = ast.parse(source, filename=file_path)

            # Extract components
            classes = []
            functions = []
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(f"import {alias.name}")

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    names = ', '.join(alias.name for alias in node.names)
                    imports.append(f"from {module} import {names}")

            # Extract top-level classes and functions
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    classes.append(self._extract_class_signature(node))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append(self._extract_function_signature(node))

            context = CodebaseContext(
                classes=classes,
                functions=functions,
                imports=list(set(imports)),  # Deduplicate
                file_path=file_path,
                total_lines=len(source.split('\n'))
            )

            # Cache the result
            self.cached_contexts[file_path] = context

            return context

        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None

    def _extract_function_signature(
        self,
        node: ast.FunctionDef
    ) -> FunctionSignature:
        """Extract function signature from AST node"""

        # Extract arguments
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)

        # Extract return type
        returns = None
        if node.returns:
            returns = ast.unparse(node.returns)

        # Extract docstring
        docstring = ast.get_docstring(node)

        return FunctionSignature(
            name=node.name,
            args=args,
            returns=returns,
            docstring=docstring,
            line_number=node.lineno,
            is_async=isinstance(node, ast.AsyncFunctionDef)
        )

    def _extract_class_signature(self, node: ast.ClassDef) -> ClassSignature:
        """Extract class signature from AST node"""

        # Extract base classes
        bases = [ast.unparse(base) for base in node.bases]

        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._extract_function_signature(item))

        # Extract docstring
        docstring = ast.get_docstring(node)

        return ClassSignature(
            name=node.name,
            bases=bases,
            methods=methods,
            docstring=docstring,
            line_number=node.lineno
        )

    def _format_contexts(
        self,
        contexts: List[CodebaseContext],
        include_bodies: bool = False
    ) -> str:
        """
        Format extracted contexts into readable string.

        Args:
            contexts: List of CodebaseContext objects
            include_bodies: If True, include full bodies (not implemented yet)

        Returns:
            Formatted context string
        """
        if not contexts:
            return "# No context available\n"

        output = ["# Codebase Context\n"]

        for context in contexts:
            # File header
            rel_path = Path(context.file_path).name
            output.append(f"\n## File: {rel_path}")
            output.append(f"Lines: {context.total_lines}\n")

            # Imports
            if context.imports:
                output.append("### Imports")
                for imp in sorted(context.imports):
                    output.append(f"{imp}")
                output.append("")

            # Classes
            if context.classes:
                output.append("### Classes")
                for cls in context.classes:
                    output.append(self._format_class(cls))
                output.append("")

            # Functions
            if context.functions:
                output.append("### Functions")
                for func in context.functions:
                    output.append(self._format_function(func))
                output.append("")

        return "\n".join(output)

    def _format_class(self, cls: ClassSignature) -> str:
        """Format class signature"""
        lines = []

        # Class definition
        bases_str = f"({', '.join(cls.bases)})" if cls.bases else ""
        lines.append(f"class {cls.name}{bases_str}:")

        # Docstring
        if cls.docstring:
            # Truncate long docstrings
            doc_preview = cls.docstring.split('\n')[0]
            if len(doc_preview) > 80:
                doc_preview = doc_preview[:77] + "..."
            lines.append(f'    """{doc_preview}"""')

        # Methods
        if cls.methods:
            for method in cls.methods:
                async_prefix = "async " if method.is_async else ""
                args_str = ", ".join(method.args)
                returns_str = f" -> {method.returns}" if method.returns else ""
                lines.append(f"    {async_prefix}def {method.name}({args_str}){returns_str}: ...")

        return "\n".join(lines)

    def _format_function(self, func: FunctionSignature) -> str:
        """Format function signature"""
        async_prefix = "async " if func.is_async else ""
        args_str = ", ".join(func.args)
        returns_str = f" -> {func.returns}" if func.returns else ""

        lines = [f"{async_prefix}def {func.name}({args_str}){returns_str}:"]

        if func.docstring:
            # Truncate long docstrings
            doc_preview = func.docstring.split('\n')[0]
            if len(doc_preview) > 80:
                doc_preview = doc_preview[:77] + "..."
            lines.append(f'    """{doc_preview}"""')
        else:
            lines.append("    ...")

        return "\n".join(lines)

    def compact_history(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> List[Dict[str, str]]:
        """
        Compact message history to fit within token limit.

        Strategy:
        1. Always keep the first message (system prompt)
        2. Always keep the last N messages (recent context)
        3. Summarize middle messages if needed

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to target (approximate)

        Returns:
            Compacted message list
        """
        if not messages:
            return []

        # Rough estimate: 1 token ≈ 4 characters
        CHARS_PER_TOKEN = 4

        # Calculate current size
        total_chars = sum(len(msg.get('content', '')) for msg in messages)
        estimated_tokens = total_chars // CHARS_PER_TOKEN

        if estimated_tokens <= max_tokens:
            # No compaction needed
            return messages

        logger.info(
            f"Compacting history: {estimated_tokens} tokens -> {max_tokens} tokens"
        )

        # Strategy: Keep first, keep last 10, summarize middle
        KEEP_LAST_N = 10

        if len(messages) <= KEEP_LAST_N + 1:
            return messages

        compacted = []

        # Keep first message (system prompt)
        compacted.append(messages[0])

        # Summarize middle messages
        middle_messages = messages[1:-KEEP_LAST_N]
        if middle_messages:
            summary = self._summarize_messages(middle_messages)
            compacted.append({
                'role': 'system',
                'content': f"[Previous conversation summary: {summary}]"
            })

        # Keep last N messages
        compacted.extend(messages[-KEEP_LAST_N:])

        # Verify compaction
        new_total_chars = sum(len(msg.get('content', '')) for msg in compacted)
        new_estimated_tokens = new_total_chars // CHARS_PER_TOKEN

        logger.info(
            f"Compaction complete: {len(messages)} messages -> {len(compacted)} messages, "
            f"{new_estimated_tokens} estimated tokens"
        )

        return compacted

    def _summarize_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Create a brief summary of messages.

        Args:
            messages: List of messages to summarize

        Returns:
            Summary string
        """
        # Count by role
        role_counts = {}
        for msg in messages:
            role = msg.get('role', 'unknown')
            role_counts[role] = role_counts.get(role, 0) + 1

        # Extract key topics (simple keyword extraction)
        all_content = " ".join(msg.get('content', '') for msg in messages)
        words = all_content.lower().split()

        # Count word frequency (excluding common words)
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'was', 'are', 'were'}
        word_freq = {}
        for word in words:
            if len(word) > 3 and word not in common_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        keywords = [word for word, _ in top_keywords]

        summary_parts = []
        summary_parts.append(f"{len(messages)} messages exchanged")

        if role_counts:
            role_summary = ", ".join(f"{count} {role}" for role, count in role_counts.items())
            summary_parts.append(f"({role_summary})")

        if keywords:
            summary_parts.append(f"Topics: {', '.join(keywords)}")

        return "; ".join(summary_parts)

    def clear_cache(self) -> None:
        """Clear the cached contexts"""
        self.cached_contexts.clear()
        logger.info("Context cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached contexts"""
        return {
            "cached_files": len(self.cached_contexts),
            "total_classes": sum(len(ctx.classes) for ctx in self.cached_contexts.values()),
            "total_functions": sum(len(ctx.functions) for ctx in self.cached_contexts.values())
        }

    def __repr__(self) -> str:
        return f"ContextManager(cached_files={len(self.cached_contexts)})"
