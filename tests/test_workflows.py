"""Tests for marqov.workflows module."""

from marqov.workflows import TemporalConfig


class TestTemporalConfig:
    """Tests for TemporalConfig."""

    def test_config_defaults(self) -> None:
        """TemporalConfig has sensible defaults."""
        config = TemporalConfig()
        assert config.host == "localhost"
        assert config.port == 7233
        assert config.namespace == "default"
        assert config.task_queue == "marqov-workflows"

    def test_config_address(self) -> None:
        """TemporalConfig builds address correctly."""
        config = TemporalConfig(host="temporal.example.com", port=7234)
        assert config.address == "temporal.example.com:7234"

    def test_config_custom_values(self) -> None:
        """TemporalConfig accepts custom values."""
        config = TemporalConfig(
            host="custom-host",
            port=9999,
            namespace="production",
            task_queue="my-queue",
        )
        assert config.host == "custom-host"
        assert config.port == 9999
        assert config.namespace == "production"
        assert config.task_queue == "my-queue"
