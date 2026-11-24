# pairwise-volumetric-cloud-reproject

## 领域

游戏、图形与多媒体

## 主题

体积云重投影累积

## 说明

固定像素哈希、运动矢量重投影与历史清零。

初始状态下 `graphics/volumetric_cloud_reproject.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
