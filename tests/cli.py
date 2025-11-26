import click
from pathlib import Path

class CkptOrString(click.ParamType):
    name = "ckpt-or-string"

    def convert(self, value, param, ctx):
        p = Path(value)

        # If the path exists AND ends with .ckpt → accept as CKPT
        if p.exists() and p.is_file() and p.suffix == ".pth":
            return p

        # If it's a path but not a .ckpt file → error
        if p.exists():
            self.fail(f"'{value}' exists but is not a .pth file", param, ctx)

        # Otherwise, treat it as a plain string
        return value


CKPT_OR_STRING = CkptOrString()


@click.command()
@click.option("--input", type=CKPT_OR_STRING, required=True)
def cli(input):
    if isinstance(input, Path):
        click.echo(f"Using CKPT file: {input}")
    else:
        click.echo(f"Using name/string: {input}")


if __name__ == "__main__":
    cli()
