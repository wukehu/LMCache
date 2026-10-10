# SPDX-License-Identifier: Apache-2.0
"""Transfer submission ownership tests without a live NIXL transport."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.transfer_channel.impl.nixl_impl import (
    NixlTransferChannelClient,
)


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_rejected_transfer_releases_handle_and_preserves_error(
    cleanup_fails: bool,
) -> None:
    """A rejected transfer has no task owner and must be released exactly once."""
    context = MagicMock()
    handle = context.agent.make_prepped_xfer.return_value
    failure = RuntimeError("remote disconnected")
    context.agent.transfer.side_effect = failure
    if cleanup_fails:
        context.agent.release_xfer_handle.side_effect = RuntimeError("cleanup failed")
    client = NixlTransferChannelClient(context, "peer", "remote-list")
    with pytest.raises(RuntimeError) as caught:
        client.submit_read([], [])
    assert caught.value is failure
    context.agent.release_xfer_handle.assert_called_once_with(handle)
    client.close()
    context.agent.release_xfer_handle.assert_called_once_with(handle)


def test_release_error_retains_task_until_release_succeeds() -> None:
    """A caller cannot recycle its read buffer before handle release succeeds."""
    context = MagicMock()
    context.agent.check_xfer_state.return_value = "ERR"
    context.agent.release_xfer_handle.side_effect = [RuntimeError("busy"), None]
    client = NixlTransferChannelClient(context, "peer", "remote-list")
    task = client.submit_read([], [])
    with pytest.raises(RuntimeError, match="busy"):
        client.query_read_status(task)
    assert client.query_read_status(task).is_finished()
    assert context.agent.release_xfer_handle.call_count == 2
    client.close()
    assert context.agent.release_xfer_handle.call_count == 2
