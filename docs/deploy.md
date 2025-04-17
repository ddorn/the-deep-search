# Deploying DS

WorkingDirectory should point to a clone of TDS's repo, and the service looks for secrets in /run/secrets/tds-env, adjust depending on your setup. Such folders are deleted on reboot.

## Frontend (streamlit)

(AI-translated)

```
# /etc/systemd/system/dsfrontend.service
[Unit]
Description=Deep Search Frontend
After=network.target
Wants=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/the-deep-search
EnvironmentFile=/run/secrets/tds-env
Restart=on-failure
RestartSec=10s
ExecStart=/usr/bin/make serve-frontend

[Install]
WantedBy=multi-user.target
```

## Backend

(AI-translated)

```
# /etc/systemd/system/dsbackend.service
[Unit]
Description=Deep Search Backend
After=network.target
Wants=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/the-deep-search
EnvironmentFile=/run/secrets/tds-env
Restart=on-failure
RestartSec=10s
ExecStart=/usr/bin/make serve-backend

[Install]
WantedBy=multi-user.target
```
