from pathlib import Path
from typing import Iterator
import sys

from litellm import batch_completion
from pydantic import BaseModel
from yaml import safe_load

import typer

from constants import SYNC_FORMAT, SYNC_PATTERN
from core_types import AssetType, DocumentStructure, PartialAsset, Task
from storage import Database
from strategies.create_structure import Section, create_document_structure
from strategies.add_sync_tokens import add_sync_tokens
from strategies.strategy import Module

PROMPT_FILE = Path(__file__).parent / "pretty_markdown.yaml"


class SectionWithContent(Section):
    section_content: str | None = None


class PrettyMarkdownConfig(BaseModel):
    sources: list[str] = []


def process_document_structure(
    structure: DocumentStructure,
    synced_text: str,
    model: str = "groq/llama-3.3-70b-versatile",
    prompt_file_path: Path | None = None
) -> str:
    """
    Process a document structure and synced text to create pretty markdown.

    Args:
        structure: The document structure containing sections
        synced_text: The raw text content of the document
        model: The LLM model to use for processing
        prompt_file_path: Path to the prompt YAML file (defaults to package prompt)

    Returns:
        The formatted markdown content
    """
    if prompt_file_path is None:
        prompt_file_path = PROMPT_FILE

    prompt = safe_load(prompt_file_path.read_text())

    flattened_sections = list(flat_section_texts(structure, synced_text))

    prompts = [
        [
            *prompt,
            dict(role="user", content=f"{title}\n{content}"),
        ]
        for title, content in flattened_sections
        if content
    ]

    responses = batch_completion(
        model=model,
        messages=prompts,
    )

    response_idx = 0
    content_parts = []
    for title, raw_content in flattened_sections:
        if not raw_content:
            content_parts.append(title)
        else:
            response = responses[response_idx].choices[0].message.content
            content_parts.append(response)
            response_idx += 1

    return "\n\n".join(content_parts)


def flat_section_texts(section: Section, text: str, depth: int = 1) -> Iterator[tuple[str, str]]:
    """Get the title and content of each section and its subsections, in a flat list."""
    section_text = text[section.start_idx:section.proper_content_end_idx]
    yield (f"{'#' * depth} {section.title}", section_text)

    for subsection in section.subsections:
        yield from flat_section_texts(subsection, text, depth + 1)


class PrettyMarkdownStrategy(Module[PrettyMarkdownConfig]):
    NAME = "pretty_markdown"
    PRIORITY = 10
    MAX_BATCH_SIZE = 1
    CONFIG_TYPE = PrettyMarkdownConfig
    INPUT_ASSET_TYPE = AssetType.STRUCTURE

    def __init__(self, config: PrettyMarkdownConfig, db: Database) -> None:
        super().__init__(config, db)
        self.prompt = safe_load(PROMPT_FILE.read_text())

    async def process_all(self, tasks: list[Task]):
        assets = self.db.get_assets([task.input_asset_id for task in tasks])

        for task, asset in zip(tasks, assets, strict=True):
            structure = DocumentStructure.model_validate_json(asset.path.read_text())
            synced_text_asset = self.db.get_assets_for_document(
                asset.document_id, AssetType.SYNCED_TEXT_FILE
            )[0]
            synced_text = synced_text_asset.path.read_text()

            content = process_document_structure(structure, synced_text)

            path = self.path_for_asset("nice_markdown", f"{asset.document_id}.md")
            path.write_text(content)

            self.db.create_asset(
                PartialAsset(
                    document_id=asset.document_id,
                    created_by_task_id=task.id,
                    type=AssetType.NICE_MARKDOWN,
                    path=path,
                )
            )


app = typer.Typer()

@app.command()
def add_structure(file: typer.FileText):
    text = file.read()
    synced_text = add_sync_tokens(text)
    structure = create_document_structure(synced_text)
    print(structure, file=sys.stderr)
    synced_formated_content = process_document_structure(structure, synced_text)
    formated_content = SYNC_PATTERN.sub("", synced_formated_content)

    print(formated_content)

if __name__ == "__main__":
    app()