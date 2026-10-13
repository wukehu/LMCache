# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest

# First Party
from lmcache.v1.distributed.read_lease import ReadClaim, ReadLeaseLedger


def test_expired_reservation_can_pin_validated_resident_object():
    ledger = ReadLeaseLedger()
    ledger.reserve(('request', 1), 1, deadline=1)
    assert not ledger.is_pinned(now=2)
    token = ledger.acquire(('request', 1), 0)
    assert token is not None
    assert ledger.is_pinned(now=1000)
    assert ledger.complete(token)
    assert not ledger.is_pinned(now=1000)


def test_partial_tp_completion_does_not_resurrect_claims():
    ledger = ReadLeaseLedger()
    owner = ('request', 1)
    ledger.reserve(owner, 4, 1)
    assert ledger.complete(ledger.acquire(owner, 0))
    assert ledger.outstanding(owner) == 3
    assert ledger.acquire(owner, 0) is None
    for rank in (1, 2, 3):
        assert ledger.complete(ledger.acquire(owner, rank))
    assert ledger.outstanding(owner) == 0
    assert ledger.forget(owner)


def test_late_callback_does_not_release_another_request():
    ledger = ReadLeaseLedger()
    ledger.reserve('A', 1, 1)
    old = ledger.acquire('A', 0)
    ledger.reserve('B', 1, 2)
    new = ledger.acquire('B', 0)
    assert ledger.complete(old)
    assert not ledger.complete(old)
    assert ledger.outstanding('B') == 1
    assert ledger.is_pinned(now=1000)
    assert ledger.complete(new)


def test_cancel_keeps_inflight_copy_pinned_and_disallows_new_readers():
    ledger = ReadLeaseLedger()
    ledger.reserve('A', 8, 1)
    token = ledger.acquire('A', 0)
    ledger.cancel('A')
    assert ledger.outstanding('A') == 1
    assert ledger.acquire('A', 1) is None
    assert not ledger.forget('A')
    assert ledger.is_pinned(now=1000)
    assert ledger.complete(token)
    assert ledger.forget('A')


def test_stale_incarnation_and_forged_token_cannot_release_new_pin():
    old, new = ReadLeaseLedger(), ReadLeaseLedger()
    for ledger in (old, new):
        ledger.reserve(('A', 1), 1, 1)
    old_token = old.acquire(('A', 1), 0)
    new_token = new.acquire(('A', 1), 0)
    assert not new.complete(old_token)
    assert not new.complete(ReadClaim(('A', 1), 0))
    assert new.is_pinned(now=1000)
    assert new.complete(new_token)


def test_pp_and_mp_owners_are_independent():
    ledger = ReadLeaseLedger()
    owners = [('request', 1, mp, pp) for mp in range(2) for pp in range(2)]
    for owner in owners:
        ledger.reserve(owner, 8, 1)
    token = ledger.acquire(owners[0], 0)
    assert ledger.complete(token)
    assert [ledger.outstanding(o) for o in owners] == [7, 8, 8, 8]


def test_invalid_counts_and_duplicate_generation_rejected():
    ledger = ReadLeaseLedger()
    with pytest.raises(ValueError):
        ledger.reserve('A', 0, 1)
    ledger.reserve('A', 1, 1)
    with pytest.raises(ValueError):
        ledger.reserve('A', 2, 2)
    assert ledger.outstanding('A') == 1


def test_abandon_is_per_reader_and_never_releases_active_copy():
    ledger = ReadLeaseLedger()
    ledger.reserve('A', 3, 1)
    token = ledger.acquire('A', 0)
    assert not ledger.abandon('A', 0)
    assert ledger.abandon('A', 1)
    assert not ledger.abandon('A', 1)
    assert ledger.outstanding('A') == 2
    assert ledger.is_pinned(1000)
    assert ledger.complete(token)
    assert ledger.abandon('A', 2)
    assert ledger.outstanding('A') == 0
    assert ledger.acquire('A', 3) is None


def test_unknown_owner_cannot_borrow_another_reservation():
    ledger = ReadLeaseLedger()
    ledger.reserve('B', 1, 10)
    assert ledger.acquire('A', 0) is None
    assert not ledger.abandon('A', 0)
    ledger.cancel('A')
    assert ledger.outstanding('B') == 1
    assert ledger.is_pinned(9)


def test_cancelled_active_owner_is_retired_after_its_callback():
    ledger = ReadLeaseLedger()
    ledger.reserve('A', 2, 1)
    token = ledger.acquire('A', 0)
    ledger.cancel('A')
    assert ledger.complete(token)
    assert ledger.outstanding('A') == 0
    assert ledger.forget('A')
    assert not ledger.complete(token)


def test_renewal_preserves_partial_reader_counts_and_completion_identity():
    ledger = ReadLeaseLedger()
    ledger.reserve('A', 3, 1)
    first = ledger.acquire('A', 0)
    assert ledger.complete(first)
    active = ledger.acquire('A', 1)
    assert ledger.renew('A', 20)
    assert ledger.outstanding('A') == 2
    assert not ledger.complete(first)
    assert ledger.acquire('A', 0) is None
    assert ledger.complete(active)
    assert ledger.abandon('A', 2)
    assert not ledger.renew('A', 100)
    assert not ledger.is_pinned(5)


def test_renewal_does_not_revive_cancelled_or_unknown_owners():
    ledger = ReadLeaseLedger()
    ledger.reserve('A', 2, 1)
    active = ledger.acquire('A', 0)
    assert ledger.cancel('A')
    assert not ledger.renew('A', 100)
    assert not ledger.renew('B', 100)
    assert ledger.outstanding('A') == 1
    assert ledger.is_pinned(1000)
    assert ledger.complete(active)
    assert not ledger.renew('A', 100)
    assert not ledger.is_pinned(0)


def test_renewal_never_shortens_deadline_or_accepts_nonfinite_values():
    ledger = ReadLeaseLedger()
    ledger.reserve('A', 1, 10)
    assert ledger.renew('A', 2)
    assert ledger.is_pinned(9)
    for deadline in (float('inf'), float('-inf'), float('nan')):
        with pytest.raises(ValueError):
            ledger.renew('A', deadline)
    assert ledger.outstanding('A') == 1
