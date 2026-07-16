from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import main


def test_managed_slot_executes_native_updater(tmp_path):
    updater = tmp_path / "bin" / "hermes-updater"
    updater.parent.mkdir()
    updater.write_bytes(b"updater")
    (tmp_path / "current.txt").write_text("1.0.0\n")

    with (
        patch("hermes_constants.get_hermes_home", return_value=tmp_path),
        patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run,
        patch.object(main, "_invalidate_update_cache") as invalidate,
    ):
        main._cmd_update_impl(SimpleNamespace(), gateway_mode=True)

    assert run.call_args.args[0] == [str(updater), "apply", "--report", "json"]
    invalidate.assert_called_once_with()


def test_package_managed_install_exits_without_mutation(tmp_path):
    with (
        patch("hermes_constants.get_hermes_home", return_value=tmp_path),
        patch("hermes_cli.config.detect_install_method", return_value="nixos"),
        patch("subprocess.run") as run,
        pytest.raises(SystemExit) as exc,
    ):
        main._cmd_update_impl(SimpleNamespace(), gateway_mode=False)

    assert exc.value.code == 1
    run.assert_not_called()


def test_checkout_update_failure_does_not_fall_back_in_place(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    failed = SimpleNamespace(success=False, errors=["dev sync failed"])
    with (
        patch("hermes_constants.get_hermes_home", return_value=tmp_path / "home"),
        patch("hermes_cli.config.detect_install_method", return_value="git"),
        patch("hermes_cli.adoption.detect_legacy_install", return_value=None),
        patch("hermes_cli.dev_update.should_use_worktree_update", return_value=True),
        patch("hermes_cli.dev_update.run_dev_update", return_value=failed),
        patch("subprocess.run") as run,
        pytest.raises(SystemExit) as exc,
    ):
        main._cmd_update_impl(SimpleNamespace(branch="main"), gateway_mode=False)

    assert exc.value.code == 1
    run.assert_not_called()
