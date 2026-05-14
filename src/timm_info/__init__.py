import math
import warnings
from typing import Tuple

import click
import timm
import torch
from torch.nn import Identity


@click.group(help='Timm model utility')
def cli():
    pass


def render_table(headers: list[str] | None, rows: list[list[str]]) -> str:
    """
    Render rows as a bordered ASCII table, optionally with a header row.

    Column widths are auto-sized to the widest cell (header or body) in each column.
    The first column is right-aligned, all remaining columns are left-aligned.
    Returns the full table as a single newline-joined string ready for ``click.echo``
    or ``print``.

    Parameters
    ----------
    headers : list of str or None
        Column header labels. When ``None``, the table is rendered without a header
        row or its separator, and the column count is derived from ``rows``.
    rows : list of list of str
        Body rows. Each inner list must have the same length as ``headers`` (or as
        the first row when ``headers`` is ``None``).

    Returns
    -------
    str
        The rendered table.
    """
    cell_rows = [[str(cell) for cell in row] for row in rows]
    if headers is not None:
        widths = [len(h) for h in headers]
    else:
        if len(cell_rows) == 0:
            return ''
        widths = [0] * len(cell_rows[0])
    for row in cell_rows:
        for i, cell in enumerate(row):
            if len(cell) > widths[i]:
                widths[i] = len(cell)

    border = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'

    def fmt(cells: list[str]) -> str:
        parts = [
            cells[i].rjust(widths[i]) if i == 0 else cells[i].ljust(widths[i])
            for i in range(len(cells))
        ]
        return '| ' + ' | '.join(parts) + ' |'

    lines = [border]
    if headers is not None:
        lines.append(fmt(headers))
        lines.append(border)
    for row in cell_rows:
        lines.append(fmt(row))
    lines.append(border)
    return '\n'.join(lines)


def _lookup_license(model_name: str) -> str:
    """
    Resolve the license string for a timm model name via its pretrained config.

    Parameters
    ----------
    model_name : str
        Full timm model name (with or without a ``.tag`` suffix).

    Returns
    -------
    str
        The license identifier, or ``"N/A"`` when none is available.
    """
    try:
        cfg = timm.get_pretrained_cfg(model_name)
        return cfg.license if cfg.license is not None else 'N/A'
    except Exception:
        return 'N/A'


def search_and_print(
    name_pattern: str,
    pretrained: bool = False,
    simple: bool = False,
    show_license: bool = False,
) -> None:
    model_names = timm.list_models(name_pattern, pretrained=pretrained)
    if len(model_names) == 0:
        if pretrained:
            click.echo(f'No pretrained models found matching the pattern {repr(name_pattern)}')
        else:
            click.echo(f'No models found matching the pattern {repr(name_pattern)}')
        return
    if simple:
        for model_name in model_names:
            if show_license:
                click.echo(f'{model_name}\t{_lookup_license(model_name)}')
            else:
                click.echo(f'{model_name}')
    else:
        header = f'Results for {repr(name_pattern)}:'
        click.echo(header)
        click.echo('-' * len(header))

        num_width = int(math.log10(len(model_names)) + 1)
        if show_license:
            rows = [
                [f'{str(i).rjust(num_width)}', model_name, _lookup_license(model_name)]
                for i, model_name in enumerate(model_names)
            ]
            click.echo(render_table(['#', 'Model', 'License'], rows))
        else:
            for i, model_name in enumerate(model_names):
                click.echo(f'{str(i).rjust(num_width)}. {model_name}')


@click.command(help='Search for timm models. Multiple patterns can be passed.')
@click.argument('name_pattern', type=str, nargs=-1)
@click.option('-p', '--pretrained', is_flag=True, help='Only show pretrained models')
@click.option('-s', '--simple', is_flag=True, help='Display results in simple format (useful for chaining I/O)')
@click.option('-l', '--license', 'show_license', is_flag=True,
              help='Also show the license for each result (tab-separated in simple mode)')
def search(
    name_pattern: list[str],
    pretrained: bool = False,
    simple: bool = False,
    show_license: bool = False,
) -> None:
    if len(name_pattern) == 0:
        click.echo('At least one pattern must be passed\n')
        exit(3)

    for pattern in name_pattern:
        search_and_print(pattern, pretrained, simple, show_license)

        if not simple:
            click.echo()


def estimate_model_size(model: torch.nn.Module) -> Tuple[int, float]:
    num_params = sum(p.numel() for p in model.parameters())
    memory_consumption = num_params * 4
    size_in_mb = memory_consumption / (1000 * 1000)
    return num_params, size_in_mb


def _replace_classifier(model, target_id: int) -> bool:
    for name, layer in reversed(list(model.named_children())):
        if id(layer) == target_id:
            setattr(model, name, Identity())
            return True
        else:
            if _replace_classifier(layer, target_id):
                return True
    return False


def replace_classifier(model, classifier):
    current_classifier_id = id(classifier)
    success = _replace_classifier(model, current_classifier_id)
    if not success:
        raise RuntimeError('Something went wrong when attempting to replace the classifier')


def process_model(model):
    classifier = model.get_classifier()

    if isinstance(classifier, Tuple):  # some timm models return multiple classifier layers
        for layer in classifier:
            replace_classifier(model, layer)
    else:
        replace_classifier(model, classifier)


def calculate_downscaling_factors(feature_model, input_size):
    """
    Calculate downscaling factors between consecutive feature extraction stages.

    Returns:
        dict: Contains downscaling factors and whether they are exact or approximate
    """
    try:
        # Create input tensor
        _, h, w = input_size
        x = torch.rand([1] + list(input_size), dtype=torch.float32)

        # Get feature maps
        with torch.no_grad():
            features = feature_model(x)

        if len(features) == 0:
            return {'downscaling_factors': 'N/A', 'downscaling_exact': 'N/A'}

        downscaling_factors = []
        exact_flags = []

        # Start with input size for first comparison
        prev_h, prev_w = h, w

        for i, feature_map in enumerate(features):
            # Calculate spatial dimensions
            feat_h, feat_w = feature_map.shape[2], feature_map.shape[3]

            # Calculate downscaling factors relative to previous stage
            scale_h = prev_h / feat_h
            scale_w = prev_w / feat_w

            # Check if downscaling is exact (integer) or approximate
            is_exact_h = abs(scale_h - round(scale_h)) < 1e-6
            is_exact_w = abs(scale_w - round(scale_w)) < 1e-6
            is_exact = is_exact_h and is_exact_w and abs(scale_h - scale_w) < 1e-6

            # Use geometric mean for non-square downscaling, then round to nearest integer
            avg_scale = (scale_h * scale_w) ** 0.5
            scale_int = round(avg_scale)

            downscaling_factors.append(f'{scale_int}x')
            exact_flags.append('exact' if is_exact else 'approx')

            # Update previous size for next iteration
            prev_h, prev_w = feat_h, feat_w

        return {
            'downscaling_factors': downscaling_factors,
            'downscaling_exact': exact_flags
        }

    except Exception:
        return {'downscaling_factors': 'N/A', 'downscaling_exact': 'N/A'}


def get_pretrained_weights_info(base_name: str) -> list[tuple[str, str]]:
    """
    Enumerate pretrained weight tags and their licenses for a base architecture.

    Parameters
    ----------
    base_name : str
        The base architecture name (e.g., ``"resnet50"``), without any ``.tag`` suffix.

    Returns
    -------
    list of tuple of (str, str)
        Each tuple is ``(tag, license)`` where ``tag`` is the portion of the full
        pretrained name after the first ``.``. ``license`` falls back to ``"N/A"``
        when the pretrained config lacks one or cannot be retrieved.
    """
    pretrained_names = timm.list_models(base_name, pretrained=True)
    results: list[tuple[str, str]] = []
    for full_name in pretrained_names:
        tag = full_name.split('.', 1)[1] if '.' in full_name else full_name
        results.append((tag, _lookup_license(full_name)))
    return results


def get_model_info(name: str) -> dict:
    can_be_pretrained = len(timm.list_models(name, pretrained=True)) > 0
    has_features = False

    # ignore warnings generated by 3rd party code (specifically torch) during model creation
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        full_model = timm.create_model(name, features_only=False, num_classes=10)

    # get number of in features for the classifier and remove classifier layers for model size estimation
    num_in_features = full_model.num_features
    process_model(full_model)

    pretrained_input_size = full_model.pretrained_cfg['input_size'] if can_be_pretrained else None

    num_params, model_size = estimate_model_size(full_model)

    del full_model

    # try and get a model with only features
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            feature_model = timm.create_model(name, features_only=True)
        has_features = True
    except (RuntimeError, AttributeError):
        pass

    if has_features:
        input_size = feature_model.pretrained_cfg.get('input_size', (3, 224, 224))
        x = torch.rand([2] + list(input_size), dtype=torch.float32)  # some models expect more than one item per batch
        with torch.no_grad():
            result = feature_model(x)
        num_feature_layers = len(result)
        num_channels_per_feature = [tensor.shape[1] for tensor in result]

        # Calculate downscaling factors
        downscaling_info = calculate_downscaling_factors(feature_model, input_size)
        downscaling_factors = downscaling_info['downscaling_factors']
        downscaling_exact = downscaling_info['downscaling_exact']
    else:
        num_feature_layers = None
        num_channels_per_feature = None
        downscaling_factors = 'N/A'
        downscaling_exact = 'N/A'

    pretrained_weights = get_pretrained_weights_info(name.split('.')[0])

    details = {
        'name': name,
        'num_params': num_params,
        'model_size': model_size,
        'classifier_num_in_features': num_in_features if num_in_features is not None else 'N/A',
        'has_features': has_features,
        'num_feature_layers': num_feature_layers if has_features else 'N/A',
        'num_channels_per_feature': num_channels_per_feature if has_features else 'N/A',
        'downscaling_factors': downscaling_factors,
        'downscaling_exact': downscaling_exact,
        'pretrained_input_size': pretrained_input_size,
        'pretrained_weights': pretrained_weights,
    }

    return details


@click.command(help='Get information about a particular timm model. Multiple names can be passed.')
@click.argument('name', type=str, nargs=-1)
def info(name: list[str]) -> None:
    if len(name) == 0:
        click.echo('At least one name must be passed\n')
        exit(3)

    encountered_error = False

    for model_name in name:
        try:
            if len(timm.list_models(model_name.split('.')[0])) == 0:
                click.echo(f'Unable to find model {repr(model_name)}')
                click.echo()
                encountered_error = True
                continue

            details = get_model_info(model_name)

        except Exception:
            click.echo(f'Something went wrong when getting information about {repr(model_name)}')
            click.echo()
            encountered_error = True
            continue

        downscaling_factors = details['downscaling_factors']
        downscaling_exact = details['downscaling_exact']
        summary_rows = [
            ['Model name', str(details['name'])],
            ['Number of params', f'{details["num_params"]:,}'],
            ['Estimated model size', f'{details["model_size"]:0.3f} MB'],
            ['Number of in features for classifier', str(details['classifier_num_in_features'])],
            ['Has extractable feature layers', str(details['has_features'])],
            ['Number of feature layers', str(details['num_feature_layers'])],
            ['Number of channels per feature', str(details['num_channels_per_feature'])],
            ['Downscaling factors per stage',
             '[' + ', '.join(downscaling_factors) + ']' if isinstance(downscaling_factors, list) else str(downscaling_factors)],
            ['Downscaling type per stage',
             '[' + ', '.join(downscaling_exact) + ']' if isinstance(downscaling_exact, list) else str(downscaling_exact)],
            ['Pretrained Input Size', str(details['pretrained_input_size'])],
        ]
        click.echo(render_table(None, summary_rows))

        weights = details['pretrained_weights']
        if len(weights) > 0:
            click.echo()
            click.echo(render_table(['Pretrained Weights', 'License'], [[t, l] for t, l in weights]))
        click.echo()

    exit(int(encountered_error))


cli.add_command(search)
cli.add_command(info)

if __name__ == '__main__':
    cli()
