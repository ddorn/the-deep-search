from pathlib import Path
from typing import Iterator

from litellm import batch_completion
from pydantic import BaseModel
from yaml import safe_load

from core_types import AssetType, DocumentStructure, PartialAsset, Task
from storage import Database
from strategies.create_structure import Section
from strategies.strategy import Module

PROMPT_FILE = Path(__file__).parent / "pretty_markdown.yaml"


class SectionWithContent(Section):
    section_content: str | None = None


class PrettyMarkdownConfig(BaseModel):
    sources: list[str] = []


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
            syncted_text_asset = self.db.get_assets_for_document(
                asset.document_id, AssetType.SYNCED_TEXT_FILE
            )[0]
            syncted_text = syncted_text_asset.path.read_text()

            flattened_sections = list(self.flat_section_texts(structure, syncted_text))

            prompts = [
                [
                    *self.prompt,
                    dict(role="user", content=f"{title}\n{content}"),
                ]
                for title, content in flattened_sections
                if content
            ]

            responses = batch_completion(
                model="groq/llama-3.3-70b-versatile",
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

            content = "\n\n".join(content_parts)

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

    def flat_section_texts(self, section: Section, text: str, depth: int = 1) -> Iterator[tuple[str, str]]:
        """Get the title and content of each section and its subsections, in a flat list."""

        section_text = text[section.start_idx:section.proper_content_end_idx]
        yield (f"{'#' * depth} {section.title}", section_text)

        for subsection in section.subsections:
            yield from self.flat_section_texts(subsection, text, depth + 1)
