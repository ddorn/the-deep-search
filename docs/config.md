# Configuration

TDS looks for a configuration file in ~/.config/the-deep-search/config.yaml

## Sample Configuration File

```yaml
sources:
  Notes:
    type: local-files
    args:
      path: ~/Documents/Notes/
```

## Sources

### Local files

Name: `local-files`
Status: **stable**

#### Arguments

- `path`: path to the files
- `ignore`: gitignore syntax to exclude files from processing

### Podcast (RSS)

Name: `rss-podcast`
Status: **unstable**

#### Arguments

- `feed`: URL of the RSS feed
- `after`: only take podcasts after X date
