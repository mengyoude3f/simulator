"""KVPoll state machine and handshake AND-gate (FINDINGS §二/§三)."""

from __future__ import annotations

import pytest

from pd_throughput_simulator_v5.kv_transfer import KVPoll, KVTransferSession


def test_initial_states():
    s = KVTransferSession("r")
    assert s.p_state == KVPoll.BOOTSTRAPPING
    assert s.d_state == KVPoll.BOOTSTRAPPING
    assert not s.p_bootstrap_done
    assert not s.d_handshake_done


def test_send_metadata_requires_decode_handshake():
    s = KVTransferSession("r")
    with pytest.raises(ValueError):
        s.d_send_metadata()  # before receiver.init


def test_decode_metadata_flips_prefill_to_waiting():
    s = KVTransferSession("r")
    s.d_receiver_init()
    assert s.d_handshake_done
    assert not s.p_bootstrap_done  # prefill still Bootstrapping
    s.d_send_metadata()
    assert s.d_metadata_sent
    assert s.p_bootstrap_done  # decode's send_metadata is the trigger


def test_chunk_send_requires_only_metadata():
    # A chunk may fly before the whole prefill finishes (SGLang streams each
    # finished chunk); the gate is decode's destination metadata.
    s = KVTransferSession("r")
    s.d_receiver_init()
    assert not s.p_can_send()  # metadata not sent yet
    with pytest.raises(ValueError):
        s.p_start_chunk_send(1024, is_last=False)
    s.d_send_metadata()
    assert s.p_can_send()  # forward need NOT be done for a mid-prefill chunk


def test_success_only_after_final_chunk_lands():
    s = KVTransferSession("r")
    s.d_receiver_init()
    s.d_send_metadata()

    s.p_start_chunk_send(1000, is_last=False)  # chunk 1
    assert s.p_state == KVPoll.TRANSFERRING
    assert s.d_state == KVPoll.TRANSFERRING
    assert not s.complete_chunk_transfer()  # landed, but no final chunk yet
    assert not s.p_transfer_succeeded

    s.p_mark_forward_done()
    s.p_start_chunk_send(500, is_last=True)  # chunk 2 (final)
    assert not s.p_transfer_succeeded  # submitted, not landed
    assert s.complete_chunk_transfer()  # final chunk lands → SUCCESS
    assert s.p_transfer_succeeded
    assert s.d_transfer_succeeded
    assert s.transfer_bytes == 1500  # cumulative over chunks


def test_final_chunk_landing_out_of_order_still_waits():
    # Final chunk submitted while an earlier chunk is still in flight: SUCCESS
    # must wait for *all* outstanding chunks.
    s = KVTransferSession("r")
    s.d_receiver_init()
    s.d_send_metadata()
    s.p_start_chunk_send(1000, is_last=False)
    s.p_start_chunk_send(500, is_last=True)
    assert not s.complete_chunk_transfer()  # one of two landed
    assert not s.p_transfer_succeeded
    assert s.complete_chunk_transfer()  # both landed → SUCCESS
    assert s.p_transfer_succeeded


def test_cannot_send_after_final_chunk():
    s = KVTransferSession("r")
    s.d_receiver_init()
    s.d_send_metadata()
    s.p_start_chunk_send(100, is_last=True)
    with pytest.raises(ValueError, match="final chunk"):
        s.p_start_chunk_send(100, is_last=False)


def test_states_are_monotonic():
    s = KVTransferSession("r")
    s.d_receiver_init()
    s.d_send_metadata()
    s.p_mark_forward_done()
    s.p_start_chunk_send(1, is_last=True)
    s.complete_chunk_transfer()
    # re-issuing an earlier transition never lowers the state
    s.d_receiver_init()
    assert s.d_state == KVPoll.SUCCESS
    assert s.p_state == KVPoll.SUCCESS
