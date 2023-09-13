from typing import List

import pytest

from utils import call_timminfo


def test_help():
    call_timminfo(['search', '--help'])


@pytest.mark.parametrize('name', [['""'], ['xception*'], ['resnet*', 'xception', 'resnet*']])
@pytest.mark.parametrize('pretrained', [True, False])
@pytest.mark.parametrize('simple', [True, False])
def test_search_for_name(name: List[str], pretrained: bool, simple: bool):
    args = ['search'] + name
    if pretrained:
        args.append('-p')
    if simple:
        args.append('-s')
    call_timminfo()
