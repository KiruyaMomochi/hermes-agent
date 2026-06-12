"""Regression test for #25676 — nested gateway.streaming config must be loaded."""
from pathlib import Path
from unittest.mock import patch, MagicMock



def _load_with_yaml_dict(yaml_dict: dict):
    """Patch filesystem so load_gateway_config() sees *yaml_dict* as config.yaml."""
    from gateway.config import load_gateway_config

    fake_home = Path("/tmp/fake_hermes_home_25676")

    def fake_exists(self):
        return str(self).endswith("config.yaml")

    with patch("gateway.config.get_hermes_home", return_value=fake_home), \
         patch.object(Path, "exists", fake_exists), \
         patch("builtins.open", create=True) as mock_file:
        mock_file.return_value.__enter__ = lambda s: s
        mock_file.return_value.__exit__ = MagicMock(return_value=False)
        with patch("yaml.safe_load", return_value=yaml_dict):
            return load_gateway_config()


class TestStreamingConfigNested:
    def test_top_level_streaming(self):
        cfg = _load_with_yaml_dict({"streaming": {"enabled": True, "transport": "draft"}})
        assert cfg.streaming.enabled is True
        assert cfg.streaming.transport == "draft"

    def test_nested_gateway_streaming(self):
        """Regression for #25676."""
        cfg = _load_with_yaml_dict({"gateway": {"streaming": {"enabled": True, "transport": "draft"}}})
        assert cfg.streaming.enabled is True
        assert cfg.streaming.transport == "draft"

    def test_top_level_takes_precedence(self):
        cfg = _load_with_yaml_dict({
            "streaming": {"enabled": True, "transport": "edit"},
            "gateway": {"streaming": {"enabled": False, "transport": "draft"}},
        })
        assert cfg.streaming.enabled is True
        assert cfg.streaming.transport == "edit"

    def test_drain_timeout_defaults_to_existing_value(self):
        cfg = _load_with_yaml_dict({})
        assert cfg.streaming.drain_timeout_seconds == 30.0

    def test_nested_gateway_streaming_drain_timeout(self):
        cfg = _load_with_yaml_dict({"gateway": {"streaming": {"drain_timeout_seconds": 12.5}}})
        assert cfg.streaming.drain_timeout_seconds == 12.5

    def test_invalid_streaming_drain_timeout_falls_back(self):
        cfg = _load_with_yaml_dict({"gateway": {"streaming": {"drain_timeout_seconds": 0.25}}})
        assert cfg.streaming.drain_timeout_seconds == 30.0


class TestInboundTimestampPrefixConfig:
    def test_min_gap_defaults_to_existing_value(self):
        cfg = _load_with_yaml_dict({})
        assert cfg.inbound_timestamp_prefix.min_gap_seconds == 60.0

    def test_nested_gateway_inbound_timestamp_min_gap(self):
        cfg = _load_with_yaml_dict({"gateway": {"inbound_timestamp_prefix": {"min_gap_seconds": 180.0}}})
        assert cfg.inbound_timestamp_prefix.min_gap_seconds == 180.0
