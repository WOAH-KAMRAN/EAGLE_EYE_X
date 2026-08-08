# Lightweight YAML configuration loader for Eagle Eye X.
#
# Tracking scripts define their constants as module-level defaults and then call
# apply_overrides(globals(), ...) to layer configuration files on top of them.
# Layering order (later wins):
#   1. built-in script defaults
#   2. <repo>/configs/*.yaml
#   3. <repo>/configs/local/*.yaml   (per-developer overrides, gitignored)
#   4. an explicit file passed via --config <path>
#
# Only UPPER_CASE keys found in the loaded YAML are written into the namespace,
# so any key missing from config files falls back to the hardcoded default.

import os

try:
    import yaml
except ImportError:
    yaml = None


def _repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


def _deep_merge(base, extra):
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def parse_config_args(argv):
    """Extract a --config <path> argument; returns (config_path, remaining_argv)."""
    if "--config" in argv:
        i = argv.index("--config")
        if i + 1 < len(argv):
            return argv[i + 1], argv[:i] + argv[i + 2:]
    return None, argv


def _merge_file(merged, path):
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    if isinstance(data, dict):
        _deep_merge(merged, data)


def load_flat_config(config_path=None):
    """Load all config layers into one flat dict of UPPER_CASE keys."""
    merged = {}
    if yaml is None:
        print("WARNING: PyYAML not installed; using built-in defaults only.")
        return merged
    root = _repo_root()
    search = [os.path.join(root, "configs"), os.path.join(root, "configs", "local")]
    for base in search:
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            if name.endswith((".yaml", ".yml")):
                _merge_file(merged, os.path.join(base, name))
    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        _merge_file(merged, config_path)
    return merged


def apply_overrides(namespace, config_path=None):
    """Write only the UPPER_CASE keys present in the config onto `namespace`."""
    cfg = load_flat_config(config_path)
    for key, value in cfg.items():
        if isinstance(key, str) and key.isupper():
            namespace[key] = value
    return cfg
