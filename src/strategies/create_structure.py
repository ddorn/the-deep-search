import re

from litellm import completion
from pydantic import BaseModel

from core_types import AssetType, PartialAsset, Rule, Task, DocumentStructure
from strategies.strategy import Module

# PROMPT = r"""
# Extract the sections of the attached podcast transcript.
# Your output should be valid json that I can load in the following pydantic model:

# ```
# class Section(BaseModel):
#     title: str
#     start_text: str
#     subsections: list[Section]
# ```

# Detailed instructions:
# - start_text:
#     - This is critical to get right.
#     - Copy exactly the first words of the section, up to the first <sync-id> tag.
#         - Ex: if the section starts with a sync-id, output it directly: "<sync-id=\"...\">".
#         - Ex: if the text is: 'Section <sync-id="19.2">1 end here. section 2 starts here<sync-id="24.5">', you would output:
#             { "title": "Document title", "start_text": "Section <sync-id=\"19.2\">", "subsections": [{ "title": "Section 1", "start_text": "Section <sync-id=\"19.2\">", "subsections": [] }, { "title": "Section 2", "start_text": "section 2 starts here<sync-id=\"24.5\">", "subsections": []}]}
#         - Ex: if the text is 'all <sync-id="1.2"> right, ...' you should use "start_text": "all <sync-id=\"1.2\">".
#     - Copy the text and sync-ids *exactly*, token for token. Keep spaces, punctuation, capitalization, typos and all. It will be searched for an exact match with full_text.find(section.start_text).
# - title:
#     - There might already be sections in the document. If so, use them. Otherwise, come up with meaningful titles and sections.
# - subsections:
#     - Make use of nested subsections as much as needed, but most documents will need only 2 or 3 levels rarely four.

# Output instructions:
# - Output an object {} and not a list. The title of the object should be the title for the whole piece and the start_sync_id should be the start of the text.
# - Output without any formating (for example, no ```json at the beginning, no ``` at the end). Output only the json, starting with {"title": ...
# """


PROMPT = r"""
Extract the sections of the attached podcast transcript.
Your output should be valid json that I can load in the following pydantic model:

```
class Section(BaseModel):
    title: str
    start_sync_id: str
    subsections: list[Section]
```

Detailed instructions:
- start_sync_id: Copy the last sync-id at or before the start of the section.
    - Ex: if the text is: 'Section <sync-id="19.2">1 end here. section 2 starts here<sync-id="24.5">', for section 2, you should output:
        { "title": "Section 2", "start_sync_id": "19.2", "subsections": []}
        Because 19.2 is before the start of section 2, but 24.5 is already after.
- title: There might already be sections in the document. If so, use them. Otherwise, come up with meaningful titles and sections.
- subsections: Make use of nested subsections as much as needed, but most documents will need only 2 or 3 levels rarely four.

Output instructions:
- Output an object {} and not a list. The title of the object should be the title for the whole piece and the start_sync_id should be the start of the text.
- Output without any formating (for example, no ```json at the beginning, no ``` at the end). Output only the json, starting with {"title": ...
"""


class Section(BaseModel):
    title: str
    start_sync_id: str
    subsections: list["Section"]


class CreateStructureConfig(BaseModel):
    sources: list[str] = []


class CreateStructureStrategy(Module[CreateStructureConfig]):
    NAME = "create_structure"
    PRIORITY = 10
    MAX_BATCH_SIZE = 1
    CONFIG_TYPE = CreateStructureConfig

    def add_rules(self, rules: list[Rule]) -> list[Rule]:
        # Combine all sources into a single regex
        if self.config.sources:
            sources_regex = "|".join(re.escape(source) for source in self.config.sources)
            return rules + [
                Rule(
                    source=sources_regex, asset_type=AssetType.SYNCED_TEXT_FILE, strategy=self.NAME
                ),
            ]
        else:
            return rules

    async def process_all(self, tasks: list[Task]):
        assets = self.db.get_assets([task.input_asset_id for task in tasks])

        for task, asset in zip(tasks, assets, strict=True):
            text = asset.path.read_text()

            document = self.db.get_document(asset.document_id)
            if document.title:
                hint = f"\n\nHint: The document title is {document.title}."
            else:
                hint = ""

            response = completion(
                model="gemini/gemini-2.5-pro-preview-03-25",
                messages=[{"role": "system", "content": PROMPT + hint}, {"role": "user", "content": text}],
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            sections = Section.model_validate_json(content)
            document_structure = self.build_document_structure(sections, text, 0)

            path = self.path_for_asset("structure", f"{asset.document_id}.json")
            path.write_text(document_structure.model_dump_json())

            self.db.create_asset(
                PartialAsset(
                    document_id=asset.document_id,
                    created_by_task_id=task.id,
                    type=AssetType.STRUCTURE,
                    path=path,
                )
            )

    def build_document_structure(self, section: Section, syncted_text: str, text_start: int) -> DocumentStructure:
        start = syncted_text.find(section.start_sync_id)
        if start == -1:
            raise ValueError(f"Start text not found for {section.title}")

        if section.subsections:
            end = syncted_text.find(section.subsections[0].start_sync_id)
        else:
            end = len(syncted_text)

        subsections = []
        for i, subsection in enumerate(section.subsections):
            subsection_start = syncted_text.find(subsection.start_sync_id)
            if i == len(section.subsections) - 1:
                subsection_text = syncted_text[subsection_start:]
            else:
                next_start = syncted_text.find(section.subsections[i + 1].start_sync_id)
                subsection_text = syncted_text[subsection_start:next_start]

            subsections.append(self.build_document_structure(subsection, subsection_text, subsection_start + text_start))

        if section.subsections:
            subsections_end_idx = max(subsection.subsections_end_idx for subsection in subsections)
        else:
            subsections_end_idx = end + text_start

        return DocumentStructure(
            title=section.title,
            start_idx=start + text_start,
            proper_content_end_idx=end + text_start,
            subsections_end_idx=subsections_end_idx,
            subsections=subsections,
        )
