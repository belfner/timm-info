timm-info
=========

## Installation

```shell
pip install timm-info
```

## Usage

timm-info provides the CLI tool `timminfo` with commands: `search` and `info`

```
Usage: timminfo [OPTIONS] COMMAND [ARGS]...

  Timm model utility

Options:
  --help  Show this message and exit.

Commands:
  info    Get information about a particular timm model.
  search  Search for timm models.
```

### search

`timminfo search ...`

```
Usage: timminfo search [OPTIONS] [NAME_PATTERN]...

  Search for timm models. Multiple patterns can be passed.

Options:
  -p, --pretrained  Only show pretrained models
  -s, --simple      Display results in simple format (useful for chaining I/O)
  --help            Show this message and exit.
```

**Examples**


Command

```shell
timminfo search 'resnet50*' 'xception*'
```

Output

```
Results for 'resnet50*':
------------------------
0. resnet50
1. resnet50_gn
2. resnet50c
3. resnet50d
4. resnet50s
5. resnet50t

Results for 'xception*':
------------------------
0. xception41
1. xception41p
2. xception65
3. xception65p
4. xception71
```

---

Command

```shell
timminfo search -s 'resnet50*' 'xception*'
```

Output

```
resnet50
resnet50_gn
resnet50c
resnet50d
resnet50s
resnet50t

xception41
xception41p
xception65
xception65p
xception71
```

---

Command

```shell
timminfo search -p 'resnet50*' 'xception*'
```

Output

```
Results for 'resnet50*':
------------------------
 0. resnet50.a1_in1k
 1. resnet50.a1h_in1k
 2. resnet50.a2_in1k
 3. resnet50.a3_in1k
 4. resnet50.am_in1k
 5. resnet50.b1k_in1k
 6. resnet50.b2k_in1k
 7. resnet50.bt_in1k
 8. resnet50.c1_in1k
 9. resnet50.c2_in1k
10. resnet50.d_in1k
11. resnet50.fb_ssl_yfcc100m_ft_in1k
12. resnet50.fb_swsl_ig1b_ft_in1k
13. resnet50.gluon_in1k
14. resnet50.ra_in1k
15. resnet50.ram_in1k
16. resnet50.tv2_in1k
17. resnet50.tv_in1k
18. resnet50_gn.a1h_in1k
19. resnet50c.gluon_in1k
20. resnet50d.a1_in1k
21. resnet50d.a2_in1k
22. resnet50d.a3_in1k
23. resnet50d.gluon_in1k
24. resnet50d.ra2_in1k
25. resnet50s.gluon_in1k

Results for 'xception*':
------------------------
0. xception41.tf_in1k
1. xception41p.ra3_in1k
2. xception65.ra3_in1k
3. xception65.tf_in1k
4. xception65p.ra3_in1k
5. xception71.tf_in1k

```

### info

`timminfo info ...`

```
Usage: timminfo info [OPTIONS] [NAME]...

  Get information about a particular timm model. Multiple names can be passed.

Options:
  --help  Show this message and exit.
```

**Example**

Command

```shell
timminfo info convnextv2_atto efficientnet_b0
```

Output

```
Model name:                      convnextv2_atto
Number of params:                3,386,760
Estimated model size:            13.547 MB
Number of feature layers:        4
Number of channels per feature:  [40, 80, 160, 320]
Pretrained Input Size: (3, 224, 224)

Model name:                      efficientnet_b0
Number of params:                3,595,388
Estimated model size:            14.382 MB
Number of feature layers:        5
Number of channels per feature:  [16, 24, 40, 112, 320]
Pretrained Input Size: (3, 224, 224)
```