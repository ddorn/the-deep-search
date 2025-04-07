import re

import platformdirs

DIRS = platformdirs.PlatformDirs("deepsearch", ensure_exists=True)

SYNC_FORMAT = '<sync-id="{id}">'
SYNC_PATTERN = re.compile(r'<sync-id="(?P<id>[^"]+)">')
