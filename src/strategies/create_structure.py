import re

from litellm import completion
from pydantic import BaseModel

from core_types import AssetType, PartialAsset, Rule, Task
from strategies.strategy import Module

PROMPT = """
Extract the sections of the attached podcast transcript.
Your output should be valid json that I can load in the following pydantic model
  ```
  class Section(BaseModel):
      title: str
      start_sync_id: str
      subsections: list[Section]
  ```
Details:
- Pay attention to field start_sync_id. This should be the full "<sync-id="...">" where the section starts. 
    - Most of the time this should just be "start_sync_id": "<sync-id=\"...\">".
    - However, if a sections doesn't start right on a sync-id, you need to find the text directly following the sync id with markup, punctuation, spaces. This text will be put in the section before, and the current section will start just after the string you copy. It will later in used in python with full_text.find(start_text) + len(start_text) to know where the sections start, so it needs to be exact!
        - For example, if the text is "Section <sync-id="19.2">1 end here. Section 2 starts here<sync-id="24.5">", the start_sync_id should be "<sync-id=\"19.2\">1 end here."
- For the title, there might already be sections in the document. If so, use them. Otherwise, come up with meaningful titles and sections.
- Make use of nested subsections as much as needed, but most documents will need only 2 or 3 levels rarely four.
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

            response = completion(
                model="gemini/gemini-2.5-pro-preview-03-25",
                messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": text}],
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            sections = Section.model_validate_json(content)

            path = self.path_for_asset("structure", f"{asset.document_id}.json")

            # Create a new asset
            path.write_text(sections.model_dump_json())
            self.db.create_asset(
                PartialAsset(
                    document_id=asset.document_id,
                    created_by_task_id=task.id,
                    type=AssetType.STRUCTURE,
                    path=path,
                )
            )
