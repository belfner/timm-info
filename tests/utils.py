import subprocess as sp
from typing import List


def call_timminfo(args: List[str] = []) -> None:
    sp.check_call(['timminfo'] + args, stderr=sp.DEVNULL, stdout=sp.DEVNULL)
