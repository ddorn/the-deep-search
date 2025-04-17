from pathlib import Path
from typing import Iterator

from litellm import batch_completion
from pydantic import BaseModel
from yaml import safe_load

from core_types import AssetType, PartialAsset, Task
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
            structure = Section.model_validate_json(asset.path.read_text())
            synced_text_asset = self.db.get_assets_for_document(
                asset.document_id, AssetType.SYNCED_TEXT_FILE
            )[0]
            synced_text = synced_text_asset.path.read_text()

            sections = self.extract_content(structure, synced_text)
            flattened_sections = list(self.flatten_sections(sections))

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

    # Find the content for each section
    @staticmethod
    def get_section_content(section: Section, text: str) -> str:
        start_pos = text.find(section.start_sync_id)
        if start_pos == -1:
            raise ValueError(f"Start text not found for {section.title}")

        if section.subsections:
            end_pos = text.find(section.subsections[0].start_sync_id)
        else:
            end_pos = len(text)

        return text[start_pos:end_pos]

    def extract_content(self, section: Section, syncted_text: str) -> SectionWithContent:
        section_content = self.get_section_content(section, syncted_text)

        subsections = []
        for i, subsection in enumerate(section.subsections):
            if i == len(section.subsections) - 1:
                next_text = syncted_text[syncted_text.find(subsection.start_sync_id) :]
            else:
                next_start = syncted_text.find(section.subsections[i + 1].start_sync_id)
                next_text = syncted_text[syncted_text.find(subsection.start_sync_id) : next_start]

            subsections.append(self.extract_content(subsection, next_text))

        return SectionWithContent(
            title=section.title,
            section_content=section_content,
            subsections=subsections,
            start_sync_id=section.start_sync_id,
        )

    def flatten_sections(self, section: Section, depth: int = 1) -> Iterator[tuple[str, str]]:
        """Get the title and content of each section and its subsections, in a flat list."""

        yield (f"{'#' * depth} {section.title}", section.section_content)

        for subsection in section.subsections:
            yield from self.flatten_sections(subsection, depth + 1)
