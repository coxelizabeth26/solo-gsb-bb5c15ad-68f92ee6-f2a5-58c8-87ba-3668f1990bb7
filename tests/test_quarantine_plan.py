"""End-to-end tests for POST /api/analyses/{id}/quarantine-plan."""

from __future__ import annotations

import ipaddress
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from app import db


@pytest.fixture()
def client() -> TestClient:
    fd, path = tempfile.mkstemp(suffix=".db", prefix="ppa-test-q-")
    os.close(fd)
    os.remove(path)  # let sqlite create it
    db.reset_db(path)
    from app.main import app

    with TestClient(app) as c:
        yield c
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(path + suffix)
        except OSError:
            pass


def _save(client: TestClient, rules: list[dict], name: str | None = None) -> str:
    body: dict = {"rules": rules}
    if name is not None:
        body["name"] = name
    r = client.post("/api/analyze", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["persisted"] is True
    return r.json()["id"]


def _assert_target_all_deny(payload: dict, target: str) -> None:
    t = ipaddress.ip_network(target)
    for block in payload["candidate_decisions"]["blocks"]:
        assert block["action"] == "deny"
        net = ipaddress.ip_network(block["cidr"])
        assert net.subnet_of(t)
        assert block["address_count"] == net.num_addresses


# ---------------------------------------------------------------------------
def test_simple_permit_target_generates_one_deny_rule(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit", "priority": 100, "source_tag": "hq"},
    ])

    r = client.post(
        f"/api/analyses/{aid}/quarantine-plan",
        json={"cidr": "10.1.0.0/24", "rule_id_prefix": "q"},
    )
    assert r.status_code == 200, r.text
    payload = r.json()

    assert payload["generatable"] is True
    assert payload["target_cidr"] == "10.1.0.0/24"
    assert payload["ip_version"] == 4
    assert payload["rule_id_prefix"] == "q"
    assert payload["blocking"] == []
    assert payload["rules"] == [
        {
            "rule_id": "q1",
            "cidr": "10.1.0.0/24",
            "action": "deny",
            "priority": 0,
            "source_tag": "quarantine",
        }
    ]
    assert payload["converted"]["cidrs"] == ["10.1.0.0/24"]
    assert payload["converted"]["address_count"] == 256
    assert payload["converted"]["range_count"] == 1
    assert payload["already_deny"]["address_count"] == 0
    assert payload["summary"]["generated_rule_count"] == 1
    assert payload["summary"]["converted_address_count"] == 256
    assert payload["summary"]["target_address_count"] == 256

    # The recomputed semantic diff: only one permit -> deny change, in target.
    changes = payload["diff"]["changes"]
    assert [(c["change_type"], c["cidrs"]) for c in changes] == [
        ("permit_to_deny", ["10.1.0.0/24"])
    ]

    # The recomputed candidate decision map over the target is deny on every
    # disjoint block, jointly exhaustive over the target.
    _assert_target_all_deny(payload, "10.1.0.0/24")
    decisions = payload["candidate_decisions"]
    assert decisions["summary"]["deny_addresses"] == 256
    assert decisions["summary"]["permit_addresses"] == 0
    assert decisions["summary"]["unmatched_addresses"] == 0


def test_unmatched_target_generates_none_to_deny_rule(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "a", "cidr": "192.168.0.0/16", "action": "permit", "priority": 100},
    ])

    r = client.post(
        f"/api/analyses/{aid}/quarantine-plan",
        json={"cidr": "10.0.0.0/24", "rule_id_prefix": "blk"},
    )
    assert r.status_code == 200, r.text
    payload = r.json()

    assert [rule["rule_id"] for rule in payload["rules"]] == ["blk1"]
    assert payload["rules"][0]["cidr"] == "10.0.0.0/24"
    assert payload["converted"]["cidrs"] == ["10.0.0.0/24"]
    types = {c["change_type"] for c in payload["diff"]["changes"]}
    assert types == {"none_to_deny"}


def test_already_deny_addresses_generate_no_rules(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit", "priority": 100},
        {"rule_id": "b", "cidr": "10.1.0.0/16", "action": "deny", "priority": 200},
        {"rule_id": "c", "cidr": "10.1.1.0/24", "action": "permit", "priority": 50},
    ])

    r = client.post(
        f"/api/analyses/{aid}/quarantine-plan",
        json={"cidr": "10.1.0.0/16", "rule_id_prefix": "q"},
    )
    assert r.status_code == 200, r.text
    payload = r.json()

    # Only the permit /24 flips; the other 65280 addresses are already deny.
    assert [rule["cidr"] for rule in payload["rules"]] == ["10.1.1.0/24"]
    assert payload["converted"]["address_count"] == 256
    assert payload["already_deny"]["address_count"] == 65536 - 256
    assert payload["already_deny"]["range_count"] == 8
    assert payload["summary"]["target_address_count"] == 65536


def test_fully_deny_target_returns_empty_plan(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "d", "cidr": "10.1.0.0/16", "action": "deny", "priority": 0},
    ])

    r = client.post(
        f"/api/analyses/{aid}/quarantine-plan",
        json={"cidr": "10.1.5.0/24", "rule_id_prefix": "q"},
    )
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["generatable"] is True
    assert payload["rules"] == []
    assert payload["converted"]["address_count"] == 0
    assert payload["already_deny"]["cidrs"] == ["10.1.5.0/24"]
    assert payload["diff"]["changes"] == []


def test_priority_zero_permit_rule_forces_subdivision(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit", "priority": 100},
        {"rule_id": "z", "cidr": "10.1.0.0/24", "action": "permit", "priority": 0},
    ])

    r = client.post(
        f"/api/analyses/{aid}/quarantine-plan",
        json={"cidr": "10.1.0.0/24", "rule_id_prefix": "tmp"},
    )
    assert r.status_code == 200, r.text
    payload = r.json()

    assert payload["generatable"] is True
    assert [rule["cidr"] for rule in payload["rules"]] == [
        "10.1.0.0/25",
        "10.1.0.128/25",
    ]
    assert [rule["rule_id"] for rule in payload["rules"]] == ["tmp1", "tmp2"]
    assert all(rule["priority"] == 0 and rule["action"] == "deny"
               for rule in payload["rules"])
    assert payload["converted"]["cidrs"] == ["10.1.0.0/24"]
    assert payload["converted"]["address_count"] == 256


def test_subdivision_subnets_only_around_existing_nodes(client: TestClient) -> None:
    # A priority-0 permit /25 sits in the first half of the target; the
    # sibling /25 has no rule and must get one ordinary /25 deny rule.
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit", "priority": 100},
        {"rule_id": "z", "cidr": "10.1.0.0/25", "action": "permit", "priority": 0},
    ])

    r = client.post(
        f"/api/analyses/{aid}/quarantine-plan",
        json={"cidr": "10.1.0.0/24", "rule_id_prefix": "q"},
    )
    assert r.status_code == 200, r.text
    cidrs = sorted(rule["cidr"] for rule in r.json()["rules"])
    assert cidrs == ["10.1.0.0/24", "10.1.0.0/26", "10.1.0.64/26"]


def test_priority_zero_permit_single_host_is_blocking(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit", "priority": 100},
        {"rule_id": "z", "cidr": "10.1.2.3/32", "action": "permit", "priority": 0},
    ])

    r = client.post(
        f"/api/analyses/{aid}/quarantine-plan",
        json={"cidr": "10.1.0.0/16", "rule_id_prefix": "q"},
    )
    assert r.status_code == 200, r.text
    payload = r.json()

    assert payload["generatable"] is False
    assert payload["rules"] is None
    assert payload["converted"] is None
    assert payload["candidate"] is None
    assert payload["diff"] is None
    assert payload["blocking"] == [
        {
            "cidr": "10.1.2.3/32",
            "address_count": 1,
            "rule_id": "z",
            "action": "permit",
            "priority": 0,
        }
    ]
    assert payload["summary"]["blocking_range_count"] == 1
    assert payload["summary"]["blocking_address_count"] == 1
    assert payload["already_deny"]["address_count"] == 0


def test_ipv6_priority_zero_host_blocker_and_huge_counts(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "v", "cidr": "2001:db8::1/128", "action": "permit", "priority": 0},
    ])

    r = client.post(
        f"/api/analyses/{aid}/quarantine-plan",
        json={"cidr": "2001:db8::/120", "rule_id_prefix": "q"},
    )
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["ip_version"] == 6
    assert payload["generatable"] is False
    assert [b["cidr"] for b in payload["blocking"]] == ["2001:db8::1/128"]

    # A generatable IPv6 plan reports exact large address counts without
    # enumerating addresses.
    aid2 = _save(client, [
        {"rule_id": "v", "cidr": "::/0", "action": "permit", "priority": 100},
    ])
    r2 = client.post(
        f"/api/analyses/{aid2}/quarantine-plan",
        json={"cidr": "2001:db8::/32", "rule_id_prefix": "q"},
    )
    assert r2.status_code == 200, r2.text
    p2 = r2.json()
    assert p2["generatable"] is True
    assert [rule["cidr"] for rule in p2["rules"]] == ["2001:db8::/32"]
    assert p2["converted"]["address_count"] == 2**96


def test_rule_id_prefix_collision_is_400(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "q1", "cidr": "10.0.0.0/8", "action": "permit", "priority": 100},
    ])

    r = client.post(
        f"/api/analyses/{aid}/quarantine-plan",
        json={"cidr": "10.1.0.0/24", "rule_id_prefix": "q"},
    )
    assert r.status_code == 400, r.text
    payload = r.json()
    assert payload["error"] == "invalid_quarantine_plan"
    assert any("q1" in issue for issue in payload["issues"])


def test_non_colliding_prefix_succeeds_with_sequential_ids(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "q1", "cidr": "10.0.0.0/8", "action": "permit", "priority": 100},
        {"rule_id": "z", "cidr": "10.1.0.0/24", "action": "permit", "priority": 0},
    ])

    r = client.post(
        f"/api/analyses/{aid}/quarantine-plan",
        json={"cidr": "10.1.0.0/24", "rule_id_prefix": "q"},
    )
    # Generated ids are q1/q2 -> q1 collides even though two rules are needed.
    assert r.status_code == 400, r.text

    r = client.post(
        f"/api/analyses/{aid}/quarantine-plan",
        json={"cidr": "10.1.0.0/24", "rule_id_prefix": "temp_"},
    )
    assert r.status_code == 200, r.text
    assert [rule["rule_id"] for rule in r.json()["rules"]] == ["temp_1", "temp_2"]


def test_bad_cidr_is_400_like_decision_map(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit", "priority": 100},
    ])

    for bad in ("10.0.0.5/24", "10.0.0.0", "999.0.0.0/8", "2001:db8::1/32"):
        r = client.post(
            f"/api/analyses/{aid}/quarantine-plan",
            json={"cidr": bad, "rule_id_prefix": "q"},
        )
        assert r.status_code == 400, (bad, r.text)
        assert r.json()["error"] == "invalid_cidr"

    # CIDR is validated before the analysis existence is checked.
    r = client.post(
        "/api/analyses/an_0000000000000000/quarantine-plan",
        json={"cidr": "10.0.0.5/24", "rule_id_prefix": "q"},
    )
    assert r.status_code == 400


def test_unknown_analysis_is_404(client: TestClient) -> None:
    r = client.post(
        "/api/analyses/an_0000000000000000/quarantine-plan",
        json={"cidr": "10.0.0.0/24", "rule_id_prefix": "q"},
    )
    assert r.status_code == 404


def test_request_validation_422(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit", "priority": 100},
    ])
    base = f"/api/analyses/{aid}/quarantine-plan"

    assert client.post(base, json={"rule_id_prefix": "q"}).status_code == 422
    assert client.post(base, json={"cidr": "10.0.0.0/24"}).status_code == 422
    assert client.post(base, json={"cidr": "10.0.0.0/24", "rule_id_prefix": "q",
                                   "extra": 1}).status_code == 422
    assert client.post(base, json={"cidr": "10.0.0.0/24",
                                   "rule_id_prefix": "   "}).status_code == 422


def test_preview_is_not_persisted(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit", "priority": 100},
    ])

    before = client.get("/api/analyses").json()["total"]
    r = client.post(
        f"/api/analyses/{aid}/quarantine-plan",
        json={"cidr": "10.1.0.0/24", "rule_id_prefix": "q"},
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["candidate"]["persisted"] is False
    assert payload["candidate"]["id"] is None

    after = client.get("/api/analyses").json()["total"]
    assert after == before
    # The saved base version still has exactly its original rule.
    rules = client.get(f"/api/analyses/{aid}/rules").json()["rules"]
    assert [row["rule_id"] for row in rules] == ["a"]


def test_repeated_calls_are_deterministic(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit", "priority": 100},
        {"rule_id": "z", "cidr": "10.1.0.0/24", "action": "permit", "priority": 0},
    ])
    body = {"cidr": "10.1.0.0/23", "rule_id_prefix": "q"}

    r1 = client.post(f"/api/analyses/{aid}/quarantine-plan", json=body).json()
    r2 = client.post(f"/api/analyses/{aid}/quarantine-plan", json=body).json()
    assert r1["rules"] == r2["rules"]
    assert [rule["rule_id"] for rule in r1["rules"]] == ["q1", "q2", "q3"]


def test_outside_target_final_action_is_unchanged(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit", "priority": 100},
        {"rule_id": "d", "cidr": "10.2.0.0/16", "action": "deny", "priority": 100},
    ])

    r = client.post(
        f"/api/analyses/{aid}/quarantine-plan",
        json={"cidr": "10.1.0.0/16", "rule_id_prefix": "q"},
    )
    assert r.status_code == 200, r.text
    payload = r.json()

    # Every semantic change is strictly inside the target.
    target = ipaddress.ip_network("10.1.0.0/16")
    for change in payload["diff"]["changes"]:
        for cidr in change["cidrs"]:
            assert ipaddress.ip_network(cidr).subnet_of(target)

    # Sanity through the recomputed decision map: an address outside the
    # target (10.2.0.0/16 deny, 10.3.0.0/16 permit) behaves as before.
    candidate = payload["candidate"]
    from app.decision_map import build_decision_map

    deny_outside = build_decision_map(candidate, ipaddress.ip_network("10.2.0.0/16"))
    assert {b["action"] for b in deny_outside["blocks"]} == {"deny"}
    permit_outside = build_decision_map(candidate, ipaddress.ip_network("10.3.0.0/16"))
    assert {b["action"] for b in permit_outside["blocks"]} == {"permit"}


def test_priority_zero_permit_shadowed_by_deny_children_is_not_blocked(
    client: TestClient,
) -> None:
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit", "priority": 100},
        {"rule_id": "z", "cidr": "10.1.0.0/24", "action": "permit", "priority": 0},
        {"rule_id": "d1", "cidr": "10.1.0.0/25", "action": "deny", "priority": 100},
        {"rule_id": "d2", "cidr": "10.1.0.128/25", "action": "deny", "priority": 100},
    ])

    r = client.post(
        f"/api/analyses/{aid}/quarantine-plan",
        json={"cidr": "10.1.0.0/24", "rule_id_prefix": "q"},
    )
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["generatable"] is True
    assert payload["blocking"] == []
    assert payload["rules"] == []
    assert payload["already_deny"]["cidrs"] == ["10.1.0.0/24"]


def test_permit_descendant_under_deny_ancestor_is_quarantined(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit", "priority": 100},
        {"rule_id": "d", "cidr": "10.1.0.0/24", "action": "deny", "priority": 100},
        {"rule_id": "p", "cidr": "10.1.0.0/25", "action": "permit", "priority": 100},
    ])

    r = client.post(
        f"/api/analyses/{aid}/quarantine-plan",
        json={"cidr": "10.1.0.0/24", "rule_id_prefix": "q"},
    )
    assert r.status_code == 200, r.text
    payload = r.json()
    assert [rule["cidr"] for rule in payload["rules"]] == ["10.1.0.0/25"]
    assert payload["converted"]["cidrs"] == ["10.1.0.0/25"]
    assert payload["already_deny"]["cidrs"] == ["10.1.0.128/25"]


def test_subdivision_around_existing_deny_half_emits_no_redundant_rule(
    client: TestClient,
) -> None:
    # Priority-0 permit /24 split into halves; one half already has an exact
    # deny /25 winner.  Only the other (permit) half needs a rule.
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit", "priority": 100},
        {"rule_id": "z", "cidr": "10.1.0.0/24", "action": "permit", "priority": 0},
        {"rule_id": "d", "cidr": "10.1.0.0/25", "action": "deny", "priority": 100},
    ])

    r = client.post(
        f"/api/analyses/{aid}/quarantine-plan",
        json={"cidr": "10.1.0.0/24", "rule_id_prefix": "q"},
    )
    assert r.status_code == 200, r.text
    payload = r.json()
    assert [rule["cidr"] for rule in payload["rules"]] == ["10.1.0.128/25"]
    assert payload["converted"]["cidrs"] == ["10.1.0.128/25"]
    assert payload["already_deny"]["cidrs"] == ["10.1.0.0/25"]


def test_target_inside_deny_ancestor_with_permit_child(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit", "priority": 100},
        {"rule_id": "d", "cidr": "10.1.0.0/16", "action": "deny", "priority": 100},
        {"rule_id": "p", "cidr": "10.1.0.0/24", "action": "permit", "priority": 100},
    ])

    r = client.post(
        f"/api/analyses/{aid}/quarantine-plan",
        json={"cidr": "10.1.0.0/23", "rule_id_prefix": "q"},
    )
    assert r.status_code == 200, r.text
    payload = r.json()
    # /23 = permit /24 (gets a rule) + already-deny 10.1.1.0/24 (no rule).
    assert [rule["cidr"] for rule in payload["rules"]] == ["10.1.0.0/24"]
    assert payload["converted"]["cidrs"] == ["10.1.0.0/24"]
    assert payload["already_deny"]["cidrs"] == ["10.1.1.0/24"]


def test_priority_zero_deny_exact_match_generates_nothing(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit", "priority": 100},
        {"rule_id": "d", "cidr": "10.1.0.0/24", "action": "deny", "priority": 0},
    ])

    r = client.post(
        f"/api/analyses/{aid}/quarantine-plan",
        json={"cidr": "10.1.0.0/24", "rule_id_prefix": "q"},
    )
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["generatable"] is True
    assert payload["rules"] == []
    assert payload["already_deny"]["address_count"] == 256
