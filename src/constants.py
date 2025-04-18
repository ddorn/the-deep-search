import re

SYNC_FORMAT = '<sync-id="{id}">'
SYNC_PATTERN = re.compile(r'<sync-id="(?P<id>[^"]+)">')
