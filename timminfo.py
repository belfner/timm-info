import math
from typing import Tuple

import click
import timm
import torch


@click.group(help='Timm model utility')
def cli():
    pass


@click.command(help='Search for timm models')
@click.argument('name_pattern', type=str)
@click.option('-p', '--pretrained', is_flag=True, help='Only show pretrained models')
@click.option('-s', '--simple', is_flag=True, help='Display results in simple format (useful for chaining I/O')
def search(name_pattern: str, pretrained: bool = False, simple: bool = False) -> None:
    model_names = timm.list_models(name_pattern, pretrained=pretrained)
    if len(model_names) == 0:
        if pretrained:
            click.echo(f'No pretrained models found matching the pattern {repr(name_pattern)}')
        else:
            click.echo(f'No models found matching the pattern {repr(name_pattern)}')
        return
    if simple:
        for i, model_name in enumerate(model_names):
            click.echo(f'{model_name}')
    else:
        header = f'Results for {repr(name_pattern)}:'
        click.echo(header)
        click.echo('-' * len(header))

        num_width = int(math.log10(len(model_names)) + 1)
        for i, model_name in enumerate(model_names):
            click.echo(f'{str(i).rjust(num_width)}. {model_name}')


def estimate_model_size(model: torch.nn.Module) -> Tuple[int, float]:
    num_params = sum(p.numel() for p in model.parameters())
    memory_consumption = num_params * 4
    size_in_mb = memory_consumption / (1000 * 1000)
    return num_params, size_in_mb


def get_model_info(name: str) -> dict:
    can_be_pretrained = len(timm.list_models(name, pretrained=True)) > 0
    model = timm.create_model(name, features_only=True)
    input_size = model.pretrained_cfg.get('input_size', (3, 128, 128))

    x = torch.rand(input_size, dtype=torch.float32).unsqueeze(0)
    result = model(x)

    num_params, model_size = estimate_model_size(model)

    details = {
        'name': name,
        'num_params': num_params,
        'model_size': model_size,
        'num_feature_layers': len(result),
        'num_channels_per_feature': [tensor.shape[1] for tensor in result],

    }
    if not can_be_pretrained:
        return details

    details['pretrained_input_size'] = model.pretrained_cfg["input_size"]

    return details


@click.command(help='Get information about a particular timm model')
@click.argument('name', type=str)
def info(name: str) -> None:
    details = get_model_info(name)
    click.echo(f'Model name:                      {details["name"]}')
    click.echo(f'Number of params:                {details["num_params"]:,}')
    click.echo(f'Estimated model size:            {details["model_size"]:0.3f} MB')
    click.echo(f'Number of feature layers:        {details["num_feature_layers"]}')
    click.echo(f'Number of channels per feature:  {details["num_channels_per_feature"]}')
    if 'num_channels_per_feature' in details:
        click.echo(f'Pretrained Input Size: {details["pretrained_input_size"]}')
    click.echo()


cli.add_command(search)
cli.add_command(info)

if __name__ == '__main__':
    cli()
