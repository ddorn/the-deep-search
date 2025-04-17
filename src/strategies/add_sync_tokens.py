import re

from constants import SYNC_FORMAT
from core_types import AssetType, PartialAsset, Task
from strategies.strategy import Module


class AddSyncTokenStrategy(Module):
    NAME = "add_sync_tokens"
    PRIORITY = 0
    MAX_BATCH_SIZE = 10

    INPUT_ASSET_TYPE = AssetType.TEXT_FILE

    async def process_all(self, tasks: list[Task]) -> None:
        assets = self.db.get_assets([task.input_asset_id for task in tasks])

        for task, asset in zip(tasks, assets):
            text = asset.path.read_text()
            with_sync_tokens = self.add_sync_tokens(text)
            out_path = self.path_for_asset("sync_tokens", asset.path.name)
            out_path.write_text(with_sync_tokens)

            self.db.create_asset(
                PartialAsset(
                    document_id=task.document_id,
                    created_by_task_id=task.id,
                    type=AssetType.SYNCED_TEXT_FILE,
                    content=None,
                    path=out_path,
                )
            )

    def add_sync_tokens(self, text: str) -> str:
        # We want to have a sync token at most every chunk_size
        # We try try to follow this constraint, by going down this list when needed:
        # - end of a line
        # - end of a sentence
        # - end of a word
        # - or abruptly cut the text

        parts = []
        start = 0
        chunk_size = 100

        token_idx = 0
        while start < len(text):
            candidate = text[start : start + chunk_size]

            # Find the first line break
            if (line_break := candidate.find("\n")) != -1:
                token_pos = start + line_break
                end = token_pos + 1

            # Find the last sentence end
            elif matches := list(re.finditer(r"[.!?]\s+", candidate)):
                match = matches[-1]
                token_pos = start + match.start() + 1
                end = start + match.end()

            # Find the last space
            elif (space := candidate.rfind(" ")) > 0:
                token_pos = start + space
                end = start + space + 1

            # Don't add any token in the middle of words
            else:
                end = start + chunk_size
                chunk = text[start:end]
                parts.append(chunk)
                start = end
                continue

            chunk = text[start:token_pos]
            chunk_end = text[token_pos:end]

            # Don't add sync tokens after empty lines
            if re.match(r"^\s*$", chunk):
                parts.append(chunk + chunk_end)
                start = end
                continue

            sync_token = SYNC_FORMAT.format(id=token_idx)
            token_idx += 1

            part = chunk + sync_token + chunk_end
            parts.append(part)
            start = end

        return "".join(parts)
