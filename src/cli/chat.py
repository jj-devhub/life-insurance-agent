# git commit: feat(cli): add interactive Rich/Typer CLI chat interface
# Module: cli/chat
"""
Interactive CLI chat interface for the Life Insurance Support Assistant.

Built with Rich (beautiful terminal output) and Typer (CLI argument parsing).
Supports two modes:
    - Direct mode: runs the agent graph directly (no server needed)
    - API mode: connects to the FastAPI server via HTTP

Features:
    - Colored, styled output with Rich Markdown rendering
    - Shows which agent handled the query and confidence score
    - Session management with /new, /history, /clear commands
    - Thinking spinner during processing
    - Graceful keyboard interrupt handling
"""

from __future__ import annotations

import sys
import uuid

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.theme import Theme

# Custom theme for the assistant
THEME = Theme(
    {
        "user": "bold cyan",
        "assistant": "bold green",
        "system": "bold yellow",
        "error": "bold red",
        "info": "dim",
        "agent": "bold magenta",
    }
)

console = Console(theme=THEME)
app = typer.Typer(
    name="lia-cli",
    help="Life Insurance Support Assistant — Interactive CLI Chat",
    add_completion=False,
)


# --------------------------------------------------------------------------- #
# Banner
# --------------------------------------------------------------------------- #

BANNER = """
[bold green]╔══════════════════════════════════════════════════════════════╗
║          🛡️  Life Insurance Support Assistant  🛡️            ║
║                                                              ║
║  Ask me about life insurance policies, coverage, claims,     ║
║  eligibility, benefits, and more!                            ║
╚══════════════════════════════════════════════════════════════╝[/bold green]

[dim]Commands:  /help  /new  /history  /clear  /exit[/dim]
"""


# --------------------------------------------------------------------------- #
# Slash commands
# --------------------------------------------------------------------------- #

HELP_TEXT = """
**Available Commands:**

| Command    | Description                            |
|------------|----------------------------------------|
| `/help`    | Show this help message                 |
| `/new`     | Start a new conversation session       |
| `/history` | Show conversation history              |
| `/clear`   | Clear current session history          |
| `/info`    | Show current session info              |
| `/exit`    | Exit the chat                          |

**Example Questions:**
- What is term life insurance?
- How do I file a life insurance claim?
- What's the difference between term and whole life?
- Am I eligible for life insurance at age 55?
- What riders should I consider?
"""


def handle_slash_command(
    command: str,
    session_id: str,
    user_id: str,
    history: list,
) -> tuple[bool, str, list]:
    """
    Handle slash commands. Returns (should_continue, session_id, history).

    Args:
        command: The slash command (e.g., "/help").
        session_id: Current session ID.
        user_id: Current user ID.
        history: Current conversation history.

    Returns:
        Tuple of (continue_loop, updated_session_id, updated_history).
    """
    cmd = command.strip().lower()

    if cmd == "/exit" or cmd == "/quit" or cmd == "/q":
        console.print("\n[system]👋 Goodbye! Stay protected.[/system]\n")
        return False, session_id, history

    elif cmd == "/help" or cmd == "/h":
        console.print(Markdown(HELP_TEXT))
        return True, session_id, history

    elif cmd == "/new":
        new_session = str(uuid.uuid4())
        console.print(f"\n[system]🔄 New session started: {new_session[:8]}...[/system]\n")
        return True, new_session, []

    elif cmd == "/history":
        if not history:
            console.print("\n[info]No conversation history yet.[/info]\n")
        else:
            table = Table(title="Conversation History", show_lines=True)
            table.add_column("Role", style="bold", width=12)
            table.add_column("Message", ratio=1)

            for msg in history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                # Truncate long messages for display
                display = content[:200] + "..." if len(content) > 200 else content
                style = "cyan" if role == "user" else "green"
                table.add_row(role.capitalize(), display, style=style)

            console.print(table)
        return True, session_id, history

    elif cmd == "/clear":
        console.print("\n[system]🗑️  Session history cleared.[/system]\n")
        return True, session_id, []

    elif cmd == "/info":
        info_panel = Panel(
            f"[bold]Session ID:[/bold] {session_id[:8]}...\n"
            f"[bold]User ID:[/bold] {user_id}\n"
            f"[bold]Messages:[/bold] {len(history)}\n"
            f"[bold]Turns:[/bold] {len([m for m in history if m.get('role') == 'user'])}",
            title="Session Info",
            border_style="blue",
        )
        console.print(info_panel)
        return True, session_id, history

    else:
        console.print(
            f"\n[error]Unknown command: {command}. Type /help for available commands.[/error]\n"
        )
        return True, session_id, history


# --------------------------------------------------------------------------- #
# Direct mode chat (no server)
# --------------------------------------------------------------------------- #


def chat_direct(
    message: str,
    user_id: str,
    session_id: str,
    history: list,
) -> dict:
    """
    Process a message using the agent graph directly (no API server needed).

    Args:
        message: User message text.
        user_id: User identifier.
        session_id: Session identifier.
        history: Conversation history as list of BaseMessage.

    Returns:
        Result dict with response, intent, agent, confidence.
    """
    from src.agents.graph import chat as agent_chat

    return agent_chat(
        message=message,
        user_id=user_id,
        session_id=session_id,
        history=history,
    )


def chat_api(
    message: str,
    user_id: str,
    session_id: str,
    api_url: str,
) -> dict:
    """
    Process a message via the FastAPI server.

    Args:
        message: User message text.
        user_id: User identifier.
        session_id: Session identifier.
        api_url: Base URL of the FastAPI server.

    Returns:
        Result dict with response, intent, agent, confidence.
    """
    import httpx

    try:
        response = httpx.post(
            f"{api_url}/api/v1/chat",
            json={
                "message": message,
                "user_id": user_id,
                "session_id": session_id,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        return {
            "response": data.get("response", ""),
            "intent": data.get("intent", "unknown"),
            "agent": data.get("agent_used", "unknown"),
            "confidence": data.get("confidence", 0.0),
        }
    except httpx.ConnectError:
        return {
            "response": "❌ Cannot connect to API server. Is it running?\n"
            f"   Start it with: `make run-api` or `uvicorn src.api.app:app`\n"
            f"   URL: {api_url}",
            "intent": "error",
            "agent": "error",
            "confidence": 0.0,
        }
    except Exception as e:
        return {
            "response": f"❌ API error: {str(e)}",
            "intent": "error",
            "agent": "error",
            "confidence": 0.0,
        }


# --------------------------------------------------------------------------- #
# Main chat loop
# --------------------------------------------------------------------------- #


@app.command()
def main(
    user_id: str = typer.Option("default_user", "--user-id", "-u", help="User ID for memory"),
    session_id: str | None = typer.Option(None, "--session-id", "-s", help="Resume a session"),
    api_url: str | None = typer.Option(
        None,
        "--api-url",
        "-a",
        help="API server URL (e.g., http://localhost:8000). If not set, runs in direct mode.",
    ),
    direct: bool = typer.Option(True, "--direct/--api", help="Direct mode (no server) or API mode"),
):
    """
    Start the interactive Life Insurance Support Assistant chat.

    By default runs in DIRECT mode (no server needed). Use --api-url to
    connect to a running FastAPI server instead.
    """
    # Determine mode
    use_api = api_url is not None

    # Initialize session
    if session_id is None:
        session_id = str(uuid.uuid4())

    history_dicts: list[dict] = []  # For display
    history_messages: list = []  # For direct mode (BaseMessage objects)

    # Print banner
    console.print(BANNER)

    mode_label = f"API ({api_url})" if use_api else "Direct"
    console.print(
        f"[info]Mode: {mode_label} | User: {user_id} | Session: {session_id[:8]}...[/info]\n"
    )

    # Main loop
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[user]You[/user]")

            if not user_input.strip():
                continue

            # Handle slash commands
            if user_input.strip().startswith("/"):
                should_continue, session_id, history_dicts = handle_slash_command(
                    user_input.strip(), session_id, user_id, history_dicts
                )
                if not should_continue:
                    break
                continue

            # Process message with spinner
            with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
                if use_api:
                    result = chat_api(user_input, user_id, session_id, api_url)
                else:
                    result = chat_direct(user_input, user_id, session_id, history_messages)
                    # Update history_messages for direct mode context
                    from langchain_core.messages import AIMessage, HumanMessage

                    history_messages.append(HumanMessage(content=user_input))
                    history_messages.append(AIMessage(content=result["response"]))

            # Update display history
            history_dicts.append({"role": "user", "content": user_input})
            history_dicts.append({"role": "assistant", "content": result["response"]})

            # Display response
            console.print()

            # Agent info bar
            intent = result.get("intent", "unknown")
            agent = result.get("agent", "unknown")
            confidence = result.get("confidence", 0.0)
            agent_display = agent.replace("_", " ").title()
            confidence_pct = f"{confidence * 100:.0f}%"

            console.print(
                f"[info]┌─ Agent: [agent]{agent_display}[/agent] │ "
                f"Intent: {intent} │ Confidence: {confidence_pct}[/info]"
            )

            # Render response as markdown in a panel
            response_md = Markdown(result["response"])
            console.print(
                Panel(
                    response_md,
                    title="[assistant]🛡️ Assistant[/assistant]",
                    border_style="green",
                    padding=(1, 2),
                )
            )

        except KeyboardInterrupt:
            console.print("\n\n[system]👋 Interrupted. Goodbye![/system]\n")
            break
        except EOFError:
            console.print("\n\n[system]👋 Goodbye![/system]\n")
            break
        except Exception as e:
            console.print(f"\n[error]Error: {str(e)}[/error]\n")
            if "--debug" in sys.argv:
                console.print_exception()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    app()
