# Personalized Topic Tracker

The Topic Tracker is a background agent that runs once a day on a schedule. It's defined on Anthropic's Managed Agents platform — the spec includes a Claude model, a system prompt telling it how to behave, and a list of tools: web search, web fetch, and file operations.

Every day, a scheduled deployment starts a new Session in a fresh Anthropic-hosted sandbox. The Session reads topics from a config file, uses web search to find what's new, fetches the top articles, summarizes them, and writes a dated report file.

Saves me time — I get a personalized daily digest without doing anything.

## Status

Work in progress. Setup and run instructions coming as the project takes shape.