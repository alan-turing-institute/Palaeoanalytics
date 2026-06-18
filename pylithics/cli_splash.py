"""
PyLithics welcome splash shown when ``pylithics`` is run with no arguments.

Renders an ANSI Shadow logo above a hero panel and a Get-started panel
using ``rich``.  ``rich`` is imported lazily so the splash module can sit
in the package without slowing down every other CLI invocation.
"""

LOGO = r"""██████╗ ██╗   ██╗██╗     ██╗████████╗██╗  ██╗██╗ ██████╗███████╗
██╔══██╗╚██╗ ██╔╝██║     ██║╚══██╔══╝██║  ██║██║██╔════╝██╔════╝
██████╔╝ ╚████╔╝ ██║     ██║   ██║   ███████║██║██║     ███████╗
██╔═══╝   ╚██╔╝  ██║     ██║   ██║   ██╔══██║██║██║     ╚════██║
██║        ██║   ███████╗██║   ██║   ██║  ██║██║╚██████╗███████║
╚═╝        ╚═╝   ╚══════╝╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝ ╚═════╝╚══════╝"""

TAGLINE = "Automated quantitative analysis of lithic illustrations"


def _version() -> str:
    """Return the installed PyLithics version, or a fallback string."""
    try:
        from importlib.metadata import version
        return f"v{version('pylithics')}"
    except Exception:
        return "v2.0"


def print_splash() -> None:
    """Render the welcome splash to the terminal."""
    from rich.align import Align
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
    panel_width = min(98, console.width)

    hero_inner = Group(
        Align.center(Text(LOGO, style="bold cyan")),
        Text(""),
        Align.center(Text(TAGLINE, style="italic")),
    )
    hero = Panel(
        hero_inner,
        title=f"[italic]PyLithics[/]  {_version()}",
        border_style="cyan",
        padding=(1, 4),
        width=panel_width,
    )

    actions = Text.assemble(
        ("▶ Quick start\n", "bold cyan"),
        ("  pylithics --data_dir pylithics/data "
         "--meta_file pylithics/data/meta_data.csv\n\n", "white"),
        ("▶ Run sample data and visualize\n", "bold cyan"),
        ("  pylithics --data_dir pylithics/data "
         "--meta_file pylithics/data/meta_data.csv --explore\n\n", "white"),
        ("▶ Open an existing run in the browser\n", "bold cyan"),
        ("  pylithics --data_dir <path> --explore\n\n", "white"),
        ("▶ Help & docs\n", "bold cyan"),
        ("  pylithics --help    pylithics --docs\n\n", "white"),
        ("▶ GitHub\n", "bold cyan"),
        ("  github.com/alan-turing-institute/Palaeoanalytics", "white"),
    )
    body = Panel(
        actions, title="[bold]Get started[/]",
        border_style="cyan", padding=(1, 2),
        width=panel_width,
    )

    console.print()
    console.print(Align.center(hero))
    console.print(Align.center(body))
    console.print()
