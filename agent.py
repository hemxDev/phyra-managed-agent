"""
Personalized Topic Tracker — Managed Agent runner.

One run of this script:
  1. Uploads topics.txt to the Files API and mounts it into a fresh session's
     container.
  2. Creates (or reuses) an Agent config and an Environment — these are
     persisted resources, not per-run parameters, so we cache their IDs
     locally and only create them once.
  3. Starts a Session, sends a kickoff message, and streams back everything
     the agent does (thinking, tool calls, text) so you can watch it work.
  4. Once the session goes idle, downloads whatever the agent wrote under
     /mnt/session/outputs/ into your local reports/ folder.

This script runs ONE session per invocation. It does not set up a daily
schedule — that's a separate Managed Agents feature (a "deployment"). Wire
that up later once you're happy with how a single run behaves.
"""

import json
import os
import time
from datetime import date
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Configuration constants
# --------------------------------------------------------------------------
# Exact model ID string, confirmed against the current model catalog — no
# date suffix, this alias always resolves to the latest Sonnet 5 snapshot.
MODEL = "claude-sonnet-5"

# Local paths this script reads from / writes to.
TOPICS_FILE = Path("topics.txt")
REPORTS_DIR = Path("reports")

# Where the agent will find topics.txt inside its container. mount_path is
# rooted under /mnt/session/uploads/, so "/topics.txt" resolves to
# /mnt/session/uploads/topics.txt — the exact path given in SYSTEM_PROMPT.
TOPICS_MOUNT_PATH = "/topics.txt"

# Anything the agent writes under /mnt/session/outputs/ is what the Files
# API can list and download after the session ends — it's the *only*
# directory that's captured that way. Writing anywhere else (e.g. back to
# /workspace) would produce a file the agent can see but we never retrieve.
OUTPUTS_REPORT_PATH = "/mnt/session/outputs/reports/{date}.md"

# Agents and Environments are persisted, versioned resources on Anthropic's
# side — creating one on every run would leave orphaned objects behind and
# pay create-latency for nothing (see "Agent ONCE, not every run" in the
# Managed Agents docs). We create them once and cache the IDs here.
STATE_FILE = Path(".agent_state.json")

# The system prompt lives on the Agent object (not the Session). Two edits
# versus the original draft, both load-bearing:
#   - "four tools" (not three) — reading topics.txt requires the `read`
#     tool; `write` alone can only create/overwrite files, not open one.
#   - output path is under /mnt/session/outputs/ — see OUTPUTS_REPORT_PATH
#     comment above for why.
SYSTEM_PROMPT = """You are a Personalized Topic Tracker background agent. Your job is to produce a daily research digest based on the user's tracked topics.

Read the file at /mnt/session/uploads/topics.txt — each line is one topic.

You have access to four tools: file read, file write, web search, and web fetch. Use them as needed.

For each topic in topics.txt:
1. Search for significant news, developments, or updates in the last 24 hours. If timestamps are unclear, include items from the last few days and note this in the summary.
2. Fetch the most relevant articles.
3. Write a 3-paragraph summary covering what happened, how it happened, and why it matters.
4. Include source URLs at the end of each section. Only cite URLs actually returned by web fetch — never invent URLs.
5. If a topic has no significant news, say so in one line — do not invent content to fill space.

Write the output as a markdown file at /mnt/session/outputs/reports/YYYY-MM-DD.md where YYYY-MM-DD is today's date (you will be told today's date in the first user message). Structure:
- H1 heading with today's date
- One H2 section per topic
- Under each H2: the summary, then a "Sources:" line with URLs
"""


# --------------------------------------------------------------------------
# Local state: cache Agent ID / Environment ID across runs
# --------------------------------------------------------------------------
def load_state() -> dict:
    """Read the cached agent_id / environment_id, if this script has run before."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict) -> None:
    """Persist agent_id / environment_id so the next run reuses them instead
    of creating duplicate Agent/Environment objects. Delete this file to
    force fresh resources (e.g. after editing SYSTEM_PROMPT below)."""
    STATE_FILE.write_text(json.dumps(state, indent=2))


# --------------------------------------------------------------------------
# One-time setup: Environment (the sandbox template) and Agent (the config)
# --------------------------------------------------------------------------
def get_or_create_environment(client: anthropic.Anthropic, state: dict) -> str:
    """Environments are reusable sandbox templates. web_search / web_fetch,
    as part of the built-in agent toolset, execute inside the session's
    container — so the container needs real network egress. We use
    unrestricted networking for simplicity; switch to `limited` with
    allowed_hosts if you want to lock this down later."""
    if "environment_id" in state:
        return state["environment_id"]

    environment = client.beta.environments.create(
        name="topic-tracker-env",
        config={
            "type": "cloud",
            "networking": {"type": "unrestricted"},
        },
    )
    state["environment_id"] = environment.id
    save_state(state)
    print(f"Created environment: {environment.id}")
    return environment.id


def get_or_create_agent(client: anthropic.Anthropic, state: dict) -> str:
    """The Agent object holds model/system/tools — never the Session. We
    enable exactly four tools out of the built-in toolset (read, write,
    web_search, web_fetch) by flipping default_config off and opting each
    one in individually, so bash/edit/glob/grep stay disabled — the agent
    has no shell access, matching "no MCP, no custom tools" from the
    design and the system prompt's stated four-tool surface."""
    if "agent_id" in state:
        return state["agent_id"]

    agent = client.beta.agents.create(
        name="Personalized Topic Tracker",
        model=MODEL,
        system=SYSTEM_PROMPT,
        tools=[
            {
                "type": "agent_toolset_20260401",
                "default_config": {"enabled": False},
                "configs": [
                    {"name": "read", "enabled": True},
                    {"name": "write", "enabled": True},
                    {"name": "web_search", "enabled": True},
                    {"name": "web_fetch", "enabled": True},
                ],
            }
        ],
    )
    state["agent_id"] = agent.id
    save_state(state)
    print(f"Created agent: {agent.id} (version {agent.version})")
    return agent.id


# --------------------------------------------------------------------------
# Per-run: upload input, create a session, stream it to completion
# --------------------------------------------------------------------------
def upload_topics_file(client: anthropic.Anthropic) -> str:
    """Upload topics.txt once per run via the Files API. The returned file_id
    is what we attach as a session resource below — it is NOT the same
    file_id the session ends up with internally (session creation makes a
    session-scoped copy), but this ID is all we need to request the mount."""
    uploaded = client.beta.files.upload(file=TOPICS_FILE)
    print(f"Uploaded {TOPICS_FILE} -> file_id {uploaded.id}")
    return uploaded.id


def create_session(client: anthropic.Anthropic, agent_id: str, environment_id: str, topics_file_id: str):
    """Start a session referencing the pre-created agent + environment.
    topics.txt is mounted read-only at TOPICS_MOUNT_PATH; session creation
    blocks until the mount is in place, so by the time this call returns the
    agent can already `read` the file."""
    session = client.beta.sessions.create(
        agent=agent_id,  # bare string = latest version of this agent
        environment_id=environment_id,
        title=f"Topic Tracker — {date.today().isoformat()}",
        resources=[
            {
                "type": "file",
                "file_id": topics_file_id,
                "mount_path": TOPICS_MOUNT_PATH,
            }
        ],
    )
    # Swap "default" below for your workspace ID if your API key isn't in
    # the org's Default workspace — this URL is just for watching along.
    print(f"Session created: {session.id}")
    print(f"Trace: https://platform.claude.com/workspaces/default/sessions/{session.id}")
    return session


def _print_event(event) -> None:
    """Render one stream event in a human-readable line. The well-documented
    event types (status transitions, agent.message, session.error) get a
    specific format; anything else falls back to a compact best-effort
    dump rather than guessing at field names that might not exist."""
    etype = event.type

    if etype.startswith("session.status_"):
        print(f"[status] {etype.removeprefix('session.status_')}")

    elif etype == "agent.message":
        for block in event.content:
            if block.type == "text":
                print(f"[agent] {block.text}")

    elif etype == "agent.thinking":
        # display defaults to "omitted" on the underlying model, so `.thinking`
        # is often empty — this just marks that a thinking step happened.
        print("[thinking] ...")

    elif etype == "agent.tool_use":
        print(f"[tool_use] {event.name} input={json.dumps(event.input)}")

    elif etype == "agent.tool_result":
        content = getattr(event, "content", None)
        text = str(content) if content is not None else "<no content field>"
        print(f"[tool_result] {text[:300]}")

    elif etype == "session.error":
        print(f"[ERROR] {getattr(event, 'message', event)}")

    else:
        # Unhandled event type (e.g. span.* progress events) — print
        # compactly rather than crash on an unexpected shape.
        print(f"[{etype}] {str(event)[:200]}")


def run_session_to_completion(client: anthropic.Anthropic, session_id: str) -> None:
    """Stream-first: open the event stream BEFORE sending the kickoff
    message. If you send first and stream second, early events (including
    fast status transitions) arrive buffered in one batch instead of live —
    see the Managed Agents client-patterns doc."""
    today = date.today().isoformat()
    kickoff_text = (
        f"Today's date is {today}. Read topics.txt and produce today's report "
        f"at /mnt/session/outputs/reports/{today}.md as described in your instructions."
    )

    with client.beta.sessions.events.stream(session_id=session_id) as stream:
        client.beta.sessions.events.send(
            session_id=session_id,
            events=[
                {
                    "type": "user.message",
                    "content": [{"type": "text", "text": kickoff_text}],
                }
            ],
        )

        for event in stream:
            _print_event(event)

            if event.type == "session.status_terminated":
                break

            if event.type == "session.status_idle":
                # requires_action means the session is waiting on a tool
                # confirmation or custom tool result. This agent has no
                # custom tools and no always_ask permission policies, so
                # that shouldn't happen here — but the check is cheap
                # insurance against a hang if the agent config changes later.
                if event.stop_reason.type != "requires_action":
                    break


# --------------------------------------------------------------------------
# After the session ends: pull the generated report back to the local disk
# --------------------------------------------------------------------------
def download_reports(client: anthropic.Anthropic, session_id: str) -> None:
    """Files written under /mnt/session/outputs/ are indexed and exposed via
    the Files API, scoped to this session. There's a brief (~1-3s) indexing
    lag right after the session goes idle, so we retry a few times before
    giving up."""
    REPORTS_DIR.mkdir(exist_ok=True)

    files = []
    for attempt in range(5):
        # scope_id is a Files-API parameter that also touches Managed Agents,
        # so it needs both betas — the SDK only auto-adds the Files header.
        files = list(
            client.beta.files.list(
                scope_id=session_id,
                betas=["managed-agents-2026-04-01"],
            )
        )
        if files:
            break
        time.sleep(1)

    if not files:
        print("No output files found — check the session trace for errors.")
        return

    files = [f for f in files if f.filename.endswith(".md")]

    for f in files:
        # Sanitize with basename: the filename comes from the sandbox and
        # should never be trusted as a safe local path.
        safe_name = os.path.basename(f.filename)
        content = client.beta.files.download(f.id)
        local_path = REPORTS_DIR / safe_name
        content.write_to_file(local_path)
        print(f"Downloaded {f.filename} -> {local_path}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> None:
    load_dotenv()  # populates ANTHROPIC_API_KEY from .env into the environment

    # Reads ANTHROPIC_API_KEY from the environment automatically — no key
    # is hardcoded here.
    client = anthropic.Anthropic()

    state = load_state()
    environment_id = get_or_create_environment(client, state)
    agent_id = get_or_create_agent(client, state)

    topics_file_id = upload_topics_file(client)
    session = create_session(client, agent_id, environment_id, topics_file_id)

    run_session_to_completion(client, session.id)
    download_reports(client, session.id)


if __name__ == "__main__":
    main()
