"""
PyLithics welcome screen, shown when ``pylithics`` starts with no arguments.

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

TAGLINE = "Quantitative analysis of lithic illustrations"


def _version() -> str:
    """Return the installed PyLithics version, or a fallback string."""
    try:
        from importlib.metadata import version
        return f"v{version('pylithics')}"
    except Exception:
        return "v2.0"


def print_splash(console=None) -> None:
    """
    Render the welcome splash to the terminal.

    Parameters
    ----------
    console : rich.console.Console, optional
        Where to print. A recording console lets the same splash be
        exported as an image for the documentation.
    """
    from rich.align import Align
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.text import Text

    console = console or Console()
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
        ("  pylithics --data_dir pylithics/data\n\n", "white"),
        ("▶ Analyse the sample data and open the dashboard\n", "bold cyan"),
        ("  pylithics --data_dir pylithics/data --explore\n\n", "white"),
        ("▶ Open a previous analysis in the browser\n", "bold cyan"),
        ("  pylithics --explore <project>/results\n\n", "white"),
        ("▶ Cut published plates into one image for each artefact\n", "bold cyan"),
        ("  pylithics-pages --data_dir <project>\n\n", "white"),
        ("▶ Help and documentation\n", "bold cyan"),
        ("  pylithics --help    pylithics --docs\n\n", "white"),
        ("▶ GitHub\n", "bold cyan"),
        ("  github.com/alan-turing-institute/Palaeoanalytics", "white"),
    )
    body = Panel(
        actions, title="[bold]Start[/]",
        border_style="cyan", padding=(1, 2),
        width=panel_width,
    )

    console.print()
    console.print(Align.center(hero))
    console.print(Align.center(body))
    console.print()
