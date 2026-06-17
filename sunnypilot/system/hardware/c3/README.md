# C3 specific hardware code

`c3` is known as `tici` (comma three). This fork targets **AGNOS 16** on Comma 3.

## Boot chain

```
/data/continue.sh
  → launch_openpilot.sh
  → launch_chffrplus.sh
  → build.py（无 prebuilt 标记时）→ manager.py
```

- 与 `spbig260427-2` 相同：无 C3 分流、无硬件检测、无 stage_firmware、无 Python 依赖 bootstrap。
- `AGNOS_VERSION` 为 `16`（`launch_env.sh`）。
