import subprocess as sp


def call_timminfo(args: list[str] | None = None) -> None:
    if args is None:
        args = []
    sp.check_call(['timminfo'] + args, stderr=sp.DEVNULL, stdout=sp.DEVNULL)
