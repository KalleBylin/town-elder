"""Tests for embedding backend selection and config."""
from __future__ import annotations

import os
import tempfile

import pytest

from town_elder.config import (
    VALID_EMBED_BACKEND_VALUES,
    ConfigError,
    EmbedBackend,
    TownElderConfig,
)
from town_elder.embeddings.backend import (
    EmbedBackendType,
    EmbedBackendUnavailableError,
    get_embed_backend_from_config,
    is_python_embed_available,
    is_rust_embed_available,
    select_embed_backend,
)


class TestEmbedBackendConfigValidation:
    """Tests for embed_backend configuration validation."""

    def test_valid_backend_values(self):
        """Test that all valid backend values are accepted."""
        with tempfile.TemporaryDirectory() as td:
            for backend in VALID_EMBED_BACKEND_VALUES:
                config = TownElderConfig(data_dir=td, embed_backend=backend)
                assert config.embed_backend == backend

    def test_invalid_backend_value_raises_error(self):
        """Test that invalid backend values raise ConfigError."""
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(ConfigError) as exc_info:
                TownElderConfig(data_dir=td, embed_backend="invalid")
            assert "Invalid embed_backend value" in str(exc_info.value)
            assert "invalid" in str(exc_info.value)

    def test_invalid_backend_value_case_sensitivity(self):
        """Test that backend values are case-insensitive in config."""
        with tempfile.TemporaryDirectory() as td:
            # Test uppercase values are normalized
            config = TownElderConfig(data_dir=td, embed_backend="PYTHON")
            assert config.embed_backend == "python"

    def test_default_backend_is_auto(self):
        """Test that default backend is auto."""
        with tempfile.TemporaryDirectory() as td:
            config = TownElderConfig(data_dir=td)
            assert config.embed_backend == "auto"

    def test_env_override(self):
        """Test that TOWN_ELDER_EMBED_BACKEND env var overrides config."""
        with tempfile.TemporaryDirectory() as td:
            # Test env override
            os.environ["TOWN_ELDER_EMBED_BACKEND"] = "rust"
            try:
                config = TownElderConfig(data_dir=td)
                assert config.embed_backend == "rust"
            finally:
                del os.environ["TOWN_ELDER_EMBED_BACKEND"]

    def test_env_override_invalid_value(self):
        """Test that invalid env value raises error."""
        with tempfile.TemporaryDirectory() as td:
            os.environ["TOWN_ELDER_EMBED_BACKEND"] = "invalid"
            try:
                with pytest.raises(ConfigError) as exc_info:
                    TownElderConfig(data_dir=td)
                assert "Invalid embed_backend value" in str(exc_info.value)
            finally:
                del os.environ["TOWN_ELDER_EMBED_BACKEND"]


class TestBackendSelection:
    """Tests for backend selection logic."""

    def test_select_python_when_required(self):
        """Test selecting python backend when available."""
        result = select_embed_backend("python", python_available=True)
        assert result == EmbedBackendType.PYTHON

    def test_select_python_unavailable_raises_error(self):
        """Test error when python backend unavailable but required."""
        with pytest.raises(EmbedBackendUnavailableError) as exc_info:
            select_embed_backend("python", python_available=False)
        assert "python" in str(exc_info.value)

    def test_select_rust_when_available(self):
        """Test selecting rust backend when available."""
        result = select_embed_backend("rust", rust_available=True)
        assert result == EmbedBackendType.RUST

    def test_select_rust_unavailable_raises_error(self):
        """Test error when rust backend unavailable but required."""
        with pytest.raises(EmbedBackendUnavailableError) as exc_info:
            select_embed_backend("rust", rust_available=False)
        assert "rust" in str(exc_info.value)
        assert "Set TE_USE_RUST_CORE=1" in str(exc_info.value)

    def test_select_auto_prefers_rust(self):
        """Test that auto mode prefers rust when available."""
        result = select_embed_backend(
            "auto", rust_available=True, python_available=True
        )
        assert result == EmbedBackendType.RUST

    def test_select_auto_fallbacks_to_python(self):
        """Test that auto mode falls back to python when rust unavailable."""
        result = select_embed_backend(
            "auto", rust_available=False, python_available=True
        )
        assert result == EmbedBackendType.PYTHON

    def test_select_auto_fallbacks_to_python_when_both_unavailable(self):
        """Test auto falls back to python even if both unavailable."""
        # python_available always returns True in real implementation,
        # but we test the logic path here
        result = select_embed_backend(
            "auto", rust_available=False, python_available=False
        )
        assert result == EmbedBackendType.PYTHON

    def test_select_invalid_value_raises_config_error(self):
        """Test that invalid config value raises ConfigError."""
        with pytest.raises(ConfigError) as exc_info:
            select_embed_backend("invalid")
        assert "Invalid embed_backend value" in str(exc_info.value)

    def test_select_case_insensitive(self):
        """Test that backend selection is case-insensitive."""
        result = select_embed_backend("PYTHON")
        assert result == EmbedBackendType.PYTHON

        result = select_embed_backend("Auto")
        assert result == EmbedBackendType.PYTHON  # auto falls back to python when rust unavailable

    def test_select_empty_defaults_to_auto(self):
        """Test that empty string defaults to auto."""
        result = select_embed_backend("")
        assert result == EmbedBackendType.PYTHON  # auto falls back to python

    def test_select_none_defaults_to_auto(self):
        """Test that None defaults to auto."""
        result = select_embed_backend(None)
        assert result == EmbedBackendType.PYTHON  # auto falls back to python


class TestGetEmbedBackendFromConfig:
    """Tests for the main entry point function."""

    def test_from_config_auto(self):
        """Test getting backend from config with auto."""
        result = get_embed_backend_from_config("auto")
        assert result in (EmbedBackendType.PYTHON, EmbedBackendType.RUST)

    def test_from_config_python(self):
        """Test getting backend from config with python."""
        result = get_embed_backend_from_config("python")
        assert result == EmbedBackendType.PYTHON

    def test_from_config_rust_unavailable(self):
        """Test getting backend from config with rust when unavailable."""
        with pytest.raises(EmbedBackendUnavailableError):
            get_embed_backend_from_config("rust")

    def test_from_config_invalid_raises_error(self):
        """Test that invalid config value raises ConfigError."""
        with pytest.raises(ConfigError):
            get_embed_backend_from_config("invalid_value")


class TestEmbedBackendAvailability:
    """Tests for backend availability checks."""

    def test_python_embed_always_available(self):
        """Test that Python embed is considered available."""
        assert is_python_embed_available() is True

    def test_rust_embed_available_check(self):
        """Test Rust embed availability check."""
        # When rust core is not enabled/available, should return False
        result = is_rust_embed_available()
        assert result is False


class TestEmbedBackendEnum:
    """Tests for EmbedBackend enum."""

    def test_enum_values(self):
        """Test that enum has correct values."""
        assert EmbedBackend.AUTO.value == "auto"
        assert EmbedBackend.PYTHON.value == "python"
        assert EmbedBackend.RUST.value == "rust"

    def test_enum_members(self):
        """Test that all expected enum members exist."""
        expected_member_count = 3
        assert len(EmbedBackend) == expected_member_count


class TestFallbackDiagnostic:
    """Tests for backend fallback diagnostic emission."""

    def test_fallback_diagnostic_emitted_once(
        self, caplog
    ) -> None:
        """Should emit fallback diagnostic once when auto falls back to Python."""
        # Reset the module-level flag
        from town_elder.embeddings import backend as embed_backend

        embed_backend.reset_fallback_diagnostic()

        # First call should emit the diagnostic
        with caplog.at_level("INFO"):
            result = embed_backend.select_embed_backend(
                "auto", rust_available=False, python_available=True
            )
        assert result == embed_backend.EmbedBackendType.PYTHON

        # Check that the diagnostic was logged
        assert any(
            "auto-selected Python (fastembed)" in record.message
            for record in caplog.records
        )

    def test_fallback_diagnostic_not_emitted_for_rust(
        self, caplog
    ) -> None:
        """Should not emit fallback diagnostic when Rust is available."""
        from town_elder.embeddings import backend as embed_backend

        embed_backend.reset_fallback_diagnostic()

        result = embed_backend.select_embed_backend(
            "auto", rust_available=True, python_available=True
        )
        assert result == embed_backend.EmbedBackendType.RUST

        # Should NOT log the fallback diagnostic
        assert not any(
            "auto-selected Python (fastembed)" in record.message
            for record in caplog.records
        )

    def test_fallback_diagnostic_not_emitted_for_explicit_python(
        self, caplog
    ) -> None:
        """Should not emit fallback diagnostic for explicit python backend."""
        from town_elder.embeddings import backend as embed_backend

        embed_backend.reset_fallback_diagnostic()

        result = embed_backend.select_embed_backend(
            "python", python_available=True
        )
        assert result == embed_backend.EmbedBackendType.PYTHON

        # Should NOT log the fallback diagnostic (it's explicit, not fallback)
        assert not any(
            "auto-selected Python (fastembed)" in record.message
            for record in caplog.records
        )

    def test_fallback_diagnostic_only_emits_once_per_process(
        self, caplog
    ) -> None:
        """Should only emit fallback diagnostic once even with multiple calls."""
        from town_elder.embeddings import backend as embed_backend

        embed_backend.reset_fallback_diagnostic()

        # First call - should emit
        embed_backend.select_embed_backend(
            "auto", rust_available=False, python_available=True
        )

        # Reset caplog to count only the second call
        caplog.clear()

        # Second call - should NOT emit again
        embed_backend.select_embed_backend(
            "auto", rust_available=False, python_available=True
        )

        # Should NOT log the fallback diagnostic on second call
        assert not any(
            "auto-selected Python (fastembed)" in record.message
            for record in caplog.records
        )
