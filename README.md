## Todos

### Necessary to release
- setup a server that runs the full pipeline
- Check periodically for new documents (possibly using one-off tasks) @diego
- fix the sync component @felix
- Nicer display of chunks (for instance, where they are in the document (section, etc.))
- Make the right pannel take 100% of the height and stay fixed when the left is scrolled
- add the urls for each source
- Tell the structure strategy the actual title of the document, and give more guidance on how to title

### Nice
- add support for epubs
- add support for pdfs
- create a ui to add sources/change config
- Make a nice README
- Sync audio when clicked on text
- Open the document when the user *hovers* the 👉 button
- Think about how to use podcast metadata on chapters in the structure strategy
- define where the config should be
- enable to have multiple instances running on the same machine?
- add search filters to search in some sources more than others (maybe default weights for sources too?)
- create a more dense transcript of the podcasts
- watch for changes in directory_source

## To review
- Different modes (main --mode desktop, main --mode server)
- By default, ignore hidden files (.*) in directory_source, and add a config argument to accept them
