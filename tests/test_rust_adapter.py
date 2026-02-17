"""Tests for the Rust adapter module."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from town_elder import rust_adapter


class TestIsRustCoreEnabled:
    """Tests for the is_rust_core_enabled function."""

    def test_default_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rust core should be disabled by default."""
        monkeypatch.delenv(rust_adapter._ENV_FLAG, raising=False)
        assert rust_adapter.is_rust_core_enabled() is False

    def test_enabled_value_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rust core should be enabled with '1'."""
        monkeypatch.setenv(rust_adapter._ENV_FLAG, "1")
        assert rust_adapter.is_rust_core_enabled() is True

    def test_enabled_value_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rust core should be enabled with 'true' (lowercase)."""
        monkeypatch.setenv(rust_adapter._ENV_FLAG, "true")
        assert rust_adapter.is_rust_core_enabled() is True

    def test_enabled_value_True(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rust core should be enabled with 'True' (capitalized)."""
        monkeypatch.setenv(rust_adapter._ENV_FLAG, "True")
        assert rust_adapter.is_rust_core_enabled() is True

    def test_enabled_value_yes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rust core should be enabled with 'yes'."""
        monkeypatch.setenv(rust_adapter._ENV_FLAG, "yes")
        assert rust_adapter.is_rust_core_enabled() is True

    def test_disabled_value_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rust core should be disabled with '0'."""
        monkeypatch.setenv(rust_adapter._ENV_FLAG, "0")
        assert rust_adapter.is_rust_core_enabled() is False

    def test_disabled_value_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rust core should be disabled with 'false'."""
        monkeypatch.setenv(rust_adapter._ENV_FLAG, "false")
        assert rust_adapter.is_rust_core_enabled() is False

    def test_disabled_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rust core should be disabled with empty string."""
        monkeypatch.setenv(rust_adapter._ENV_FLAG, "")
        assert rust_adapter.is_rust_core_enabled() is False

    def test_disabled_whitespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rust core should be disabled with whitespace string."""
        monkeypatch.setenv(rust_adapter._ENV_FLAG, "   ")
        assert rust_adapter.is_rust_core_enabled() is False


class TestGetTeCore:
    """Tests for the get_te_core function."""

    def test_flag_off_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When flag is off, should return None without checking module."""
        monkeypatch.delenv(rust_adapter._ENV_FLAG, raising=False)
        # Even if we mock _check_rust_available to return True, should still get None
        with mock.patch.object(rust_adapter, "_check_rust_available", return_value=True):
            rust_adapter._reset_module_cache()
            result = rust_adapter.get_te_core()
            assert result is None

    def test_flag_on_module_available_returns_module(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When flag is on and module available, should return module."""
        monkeypatch.setenv(rust_adapter._ENV_FLAG, "1")
        mock_module = mock.MagicMock()
        with (
            mock.patch.object(
                rust_adapter, "_check_rust_available", return_value=True
            ),
            mock.patch.dict("sys.modules", {"town_elder._te_core": mock_module}),
        ):
            rust_adapter._reset_module_cache()
            result = rust_adapter.get_te_core()
            assert result is mock_module

    def test_flag_on_module_unavailable_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When flag is on but module unavailable, should return None."""
        monkeypatch.setenv(rust_adapter._ENV_FLAG, "1")
        with mock.patch.object(
            rust_adapter, "_check_rust_available", return_value=False
        ):
            rust_adapter._reset_module_cache()
            result = rust_adapter.get_te_core()
            assert result is None


class TestGetTeCoreOrRaise:
    """Tests for the get_te_core_or_raise function."""

    def test_flag_off_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When flag is off, should raise RustExtensionNotAvailableError."""
        monkeypatch.delenv(rust_adapter._ENV_FLAG, raising=False)
        rust_adapter._reset_module_cache()
        with pytest.raises(rust_adapter.RustExtensionNotAvailableError) as exc_info:
            rust_adapter.get_te_core_or_raise()
        assert "disabled" in str(exc_info.value).lower()

    def test_flag_on_module_unavailable_raises_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When flag is on but module unavailable, should raise error."""
        monkeypatch.setenv(rust_adapter._ENV_FLAG, "1")
        with mock.patch.object(
            rust_adapter, "_check_rust_available", return_value=False
        ):
            rust_adapter._reset_module_cache()
            with pytest.raises(rust_adapter.RustExtensionNotAvailableError):
                rust_adapter.get_te_core_or_raise()

    def test_flag_on_module_available_returns_module(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When flag is on and module available, should return module."""
        monkeypatch.setenv(rust_adapter._ENV_FLAG, "1")
        mock_module = mock.MagicMock()
        with (
            mock.patch.object(
                rust_adapter, "_check_rust_available", return_value=True
            ),
            mock.patch.dict("sys.modules", {"town_elder._te_core": mock_module}),
        ):
            rust_adapter._reset_module_cache()
            result = rust_adapter.get_te_core_or_raise()
            assert result is mock_module


class TestHealthCheck:
    """Tests for the health_check function."""

    def test_returns_none_when_flag_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return None when flag is off."""
        monkeypatch.delenv(rust_adapter._ENV_FLAG, raising=False)
        rust_adapter._reset_module_cache()
        assert rust_adapter.health_check() is None


class TestVersion:
    """Tests for the version function."""

    def test_returns_none_when_flag_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return None when flag is off."""
        monkeypatch.delenv(rust_adapter._ENV_FLAG, raising=False)
        rust_adapter._reset_module_cache()
        assert rust_adapter.version() is None


class TestAdapterStatus:
    """Tests for the get_adapter_status function."""

    def test_status_disabled_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should report disabled by default."""
        monkeypatch.delenv(rust_adapter._ENV_FLAG, raising=False)
        with mock.patch.object(
            rust_adapter, "_check_rust_available", return_value=False
        ):
            rust_adapter._reset_module_cache()
            status = rust_adapter.get_adapter_status()
            assert status["rust_core_enabled"] is False
            assert status["module_available"] is False
            assert status["flag_environment"] == rust_adapter._ENV_FLAG

    def test_status_enabled_flag_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should report enabled when flag is set."""
        monkeypatch.setenv(rust_adapter._ENV_FLAG, "1")
        with mock.patch.object(
            rust_adapter, "_check_rust_available", return_value=False
        ):
            rust_adapter._reset_module_cache()
            status = rust_adapter.get_adapter_status()
            assert status["rust_core_enabled"] is True


class TestResetModuleCache:
    """Tests for the _reset_module_cache function."""

    def test_resets_cached_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should reset the internal cache state."""
        # First, set up some state
        monkeypatch.setenv(rust_adapter._ENV_FLAG, "1")
        # This should not raise even after multiple calls
        rust_adapter._reset_module_cache()
        rust_adapter._reset_module_cache()
        # Should work normally after reset
        monkeypatch.delenv(rust_adapter._ENV_FLAG, raising=False)
        rust_adapter._reset_module_cache()
        assert rust_adapter.get_te_core() is None


class TestErrorMessages:
    """Tests for error message content."""

    def test_default_error_message_mentions_build(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default error message should mention how to build the extension."""
        monkeypatch.setenv(rust_adapter._ENV_FLAG, "1")
        with mock.patch.object(
            rust_adapter, "_check_rust_available", return_value=False
        ):
            rust_adapter._reset_module_cache()
            with pytest.raises(rust_adapter.RustExtensionNotAvailableError) as exc_info:
                rust_adapter.get_te_core_or_raise()
            error_msg = str(exc_info.value)
            assert "maturin" in error_msg or "build" in error_msg.lower()

    def test_disabled_error_message_mentions_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Error message when disabled should mention the flag."""
        monkeypatch.delenv(rust_adapter._ENV_FLAG, raising=False)
        rust_adapter._reset_module_cache()
        with pytest.raises(rust_adapter.RustExtensionNotAvailableError) as exc_info:
            rust_adapter.get_te_core_or_raise()
        error_msg = str(exc_info.value)
        assert "TE_USE_RUST_CORE" in error_msg


class TestBackwardCompatibility:
    """Tests to verify backward compatibility (default behavior)."""

    def test_default_no_module_loaded(self) -> None:
        """By default (flag off), no Rust module should be loaded."""
        # Ensure environment is clean
        env_val = os.environ.get(rust_adapter._ENV_FLAG)
        if env_val is not None:
            del os.environ[rust_adapter._ENV_FLAG]
        rust_adapter._reset_module_cache()

        # Should return None, not raise
        result = rust_adapter.get_te_core()
        assert result is None

        # health_check and version should also return None
        assert rust_adapter.health_check() is None
        assert rust_adapter.version() is None
