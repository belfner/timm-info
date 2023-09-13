from typing import List

import pytest

from utils import call_timminfo


def test_help():
    call_timminfo(['info', '--help'])


@pytest.mark.parametrize('name', [['resnet50'],
                                  ['vgg11'],
                                  ['mobilenetv2_110d'],
                                  ['resnet50', 'vgg11', 'mobilenetv2_110d']])
def test_get_info(name: List[str]):
    call_timminfo(['info'] + name)
