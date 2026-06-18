"""
Throwaway demo of four PyLithics splash styles using rich.

Run:  python splash_demo.py
Delete after picking a favorite.
"""

from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

console = Console()

LOGO = r"""██████╗ ██╗   ██╗██╗     ██╗████████╗██╗  ██╗██╗ ██████╗███████╗
██╔══██╗╚██╗ ██╔╝██║     ██║╚══██╔══╝██║  ██║██║██╔════╝██╔════╝
██████╔╝ ╚████╔╝ ██║     ██║   ██║   ███████║██║██║     ███████╗
██╔═══╝   ╚██╔╝  ██║     ██║   ██║   ██╔══██║██║██║     ╚════██║
██║        ██║   ███████╗██║   ██║   ██║  ██║██║╚██████╗███████║
╚═╝        ╚═╝   ╚══════╝╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝ ╚═════╝╚══════╝"""

TAGLINE = "Automated quantitative analysis of lithic illustrations"
VERSION = "v2.0.0"


def style_1_minimal_panel():
    """Plain logo, single rounded panel underneath. Closest to Claude Code."""
    console.print()
    console.print(Text(LOGO, style="bold magenta"))
    console.print(Text(f"  {TAGLINE}  ·  {VERSION}", style="dim"))
    console.print()

    body = Text.assemble(
        ("Try it now\n", "bold cyan"),
        ("  pylithics --data_dir pylithics/data \\\n", "white"),
        ("            --meta_file pylithics/data/meta_data.csv\n\n", "white"),
        ("Already analyzed? Explore in the browser\n", "bold cyan"),
        ("  pylithics --data_dir <path> --explore\n\n", "white"),
        ("Help: ", "dim"), ("pylithics --help", "white"),
        ("    ", "dim"),
        ("Docs: ", "dim"), ("pylithics --docs", "white"),
    )
    console.print(Panel(body, border_style="magenta", padding=(1, 2)))


def style_2_hero_panel():
    """Logo inside a bold-bordered hero panel with version as title."""
    inner = Group(
        Align.center(Text(LOGO, style="bold cyan")),
        Text(""),
        Align.center(Text(TAGLINE, style="italic")),
        Text(""),
        Align.center(Text("Quick start", style="bold yellow")),
        Text(""),
        Align.center(Text(
            "pylithics --data_dir pylithics/data \\", style="white"
        )),
        Align.center(Text(
            "          --meta_file pylithics/data/meta_data.csv",
            style="white",
        )),
        Text(""),
        Align.center(Text("pylithics --help    pylithics --docs",
                          style="dim")),
    )
    console.print()
    console.print(Panel(inner, title=f"[bold]PyLithics[/]  {VERSION}",
                        border_style="cyan", padding=(1, 4)))


def style_3_two_panel_stack():
    """Hero panel on top, structured Get-started panel underneath."""
    hero = Panel(
        Align.center(
            Group(
                Text(LOGO, style="bold green"),
                Text(""),
                Text(f"{TAGLINE}  ·  {VERSION}", style="dim italic"),
            ),
            vertical="middle",
        ),
        border_style="green",
        padding=(1, 2),
    )

    actions = Text.assemble(
        ("▶ Sample data\n", "bold green"),
        ("  pylithics --data_dir pylithics/data \\\n", "white"),
        ("            --meta_file pylithics/data/meta_data.csv\n\n", "white"),
        ("▶ Open dashboard\n", "bold green"),
        ("  pylithics --data_dir pylithics/data --explore\n\n", "white"),
        ("▶ Help & docs\n", "bold green"),
        ("  pylithics --help    pylithics --docs\n", "white"),
    )
    body = Panel(actions, title="[bold]Get started[/]",
                 border_style="green", padding=(1, 2))

    console.print()
    console.print(hero)
    console.print(body)


def style_4_card_grid():
    """Cards-in-columns: three action cards side by side under the logo."""
    console.print()
    console.print(Text(LOGO, style="bold blue"))
    console.print(Text(f"  {TAGLINE}", style="dim italic"))
    console.print(Text(f"  {VERSION}", style="dim"))
    console.print()

    sample = Panel(
        Text.assemble(
            ("Run on bundled\nsample data\n\n", "bold"),
            ("pylithics \\\n  --data_dir \\\n  pylithics/data \\\n"
             "  --meta_file ...", "white"),
        ),
        title="[bold blue]Sample[/]", border_style="blue", padding=(1, 1),
    )
    explore = Panel(
        Text.assemble(
            ("Open an existing\nrun in the browser\n\n", "bold"),
            ("pylithics \\\n  --data_dir <path> \\\n  --explore", "white"),
        ),
        title="[bold blue]Explore[/]", border_style="blue", padding=(1, 1),
    )
    help_card = Panel(
        Text.assemble(
            ("Get help / open\nthe docs site\n\n", "bold"),
            ("pylithics --help\npylithics --docs", "white"),
        ),
        title="[bold blue]Help[/]", border_style="blue", padding=(1, 1),
    )
    console.print(Columns([sample, explore, help_card], equal=True,
                          expand=True))


def style_5_combo():
    """Hero panel (style 2) above a Get-started panel (style 3)."""
    panel_width = min(98, console.width)

    hero_inner = Group(
        Align.center(Text(LOGO, style="bold cyan")),
        Text(""),
        Align.center(Text(TAGLINE, style="italic")),
    )
    hero = Panel(
        hero_inner,
        title=f"[italic]PyLithics[/]  {VERSION}",
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


def main():
    styles = [
        ("Style 1 — Minimal: plain logo + single panel", style_1_minimal_panel),
        ("Style 2 — Hero panel with logo inside", style_2_hero_panel),
        ("Style 3 — Two stacked panels", style_3_two_panel_stack),
        ("Style 4 — Logo + three action cards", style_4_card_grid),
        ("Style 5 — Combo (your pick): hero + Get started", style_5_combo),
    ]
    for label, render in styles:
        console.print()
        console.print(Rule(f"[bold]{label}[/]", style="dim"))
        render()
        console.print()
    console.print(Rule(style="dim"))
    console.print("[dim]Pick a favorite, then we'll wire it into the "
                  "pylithics CLI entry point.[/]")


if __name__ == "__main__":
    main()
