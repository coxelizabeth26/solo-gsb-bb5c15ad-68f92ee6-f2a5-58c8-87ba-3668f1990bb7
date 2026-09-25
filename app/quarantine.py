"""Temporary quarantine ("deny-only") rule plan on top of one saved version.

Given a *strict* target CIDR and a new-rule id prefix, this module derives a
**deterministic** set of brand-new ``deny`` rules (every one with
``priority = 0``) that, when added to the saved version, make **every address
inside the target resolve to deny** while leaving the final action of every
address outside the target exactly as it was.  The plan is a preview only:
the analyzer is re-run in memory, nothing is persisted and no analysis
version is created.

Generation semantics
--------------------
Forwarding follows the engine's rules: a longer prefix always wins, and
exact-CIDR ties are resolved by priority (smaller number first) and then
submission order.  Every added rule has priority 0 and is appended after the
base rules, so:

* an added rule whose CIDR is *longer* than every existing rule covering the
  address wins by longest-prefix match;
* an added rule on the *exact same CIDR* as an existing rule wins exactly
  when the existing winner's priority is greater than 0 (priority 0 beats it);
  an existing **priority-0** winner keeps the CIDR (the new rule, appended
  later, loses the same-priority tie).

The target is therefore walked geometrically over the base version's winner
trie:

* a region already resolving to deny needs no rule;
* a region covered only by a permit rule with priority > 0 (or by no rule at
  all) is handled by one deny rule at the region's own CIDR;
* an exact-match existing **priority-0 permit** winner cannot be beaten at
  that CIDR, so the region is *subdivided* into its two child prefixes and
  each child is handled recursively — longer-prefix added rules then beat the
  priority-0 rule;
* subdivision that reaches a single host (``/32`` IPv4, ``/128`` IPv6) still
  blocked by an exact priority-0 permit rule cannot win at all: that address
  is reported as an un-generatable *blocking* range.

The walk starts with the action of the longest existing winner rule that
encloses the whole target (``none`` when nothing encloses it), and longer
prefixes inside the target always override that background — so only the
target's own permit/unmatched surface ever generates rules and the final
action outside the target is untouched.

Already-deny addresses never produce a rule.  The candidate is then rebuilt
and **re-verified** with the existing analysis machinery: the candidate rules
are re-run through the LPM analyzer (the full unsaved ``candidate`` result),
and a fast action-only target partition (the same winner-trie geometry as the
decision map, without the quadratic match chains) confirms the target resolves
to deny on every address.  The converted surface and the returned semantic
``diff`` are produced by overlaying the base and candidate target partitions
with the same geometry as the version semantic diff, restricted to the
target (addresses outside it cannot change because every generated rule is
contained in the target).  On success only ``permit → deny`` /
``none → deny`` surfaces change and the target is deny end-to-end.
"""

from __future__ import annotations

import bisect
import ipaddress
from typing import Any, NamedTuple

from .analyzer import _sort_networks, _subtract, analyze_rules
from .diff import CHANGE_TYPE_ORDER, Action, _overlay, change_type_name
from .simulate import rebuild_input_rules

Network = Any  # ipaddress._BaseNetwork

#: ``source_tag`` stamped on every generated temporary quarantine rule.
QUARANTINE_SOURCE_TAG = "quarantine"

#: Every generated rule is deny at the highest same-CIDR priority.
QUARANTINE_PRIORITY = 0
QUARANTINE_ACTION = "deny"


class QuarantinePlanError(Exception):
    """The plan cannot be produced as requested (e.g. generated ids collide).

    ``issues`` lists every problem found in deterministic order.  Bad target
    CIDRs are reported separately as ``invalid_cidr`` via
    :func:`app.decision_map.parse_query_cidr`.
    """

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("; ".join(issues))


# ---------------------------------------------------------------------------
# Deterministic rule derivation
# ---------------------------------------------------------------------------
class _WinnerNode(NamedTuple):
    network: Network
    action: str
    priority: int
    rule_id: str


def _winner_nodes(base: dict[str, Any]) -> dict[int, list[_WinnerNode]]:
    """Winner rules of the stored version grouped by IP family."""
    nodes: dict[int, list[_WinnerNode]] = {4: [], 6: []}
    for rule in base.get("rules", []):
        if rule.get("role") != "winner":
            continue
        node = _WinnerNode(
            network=ipaddress.ip_network(rule["cidr"]),
            action=str(rule["action"]),
            priority=int(rule["priority"]),
            rule_id=str(rule["rule_id"]),
        )
        nodes[node.network.version].append(node)
    return nodes


class _WinnerTrie:
    """Containment trie of one family's winner rules, built once.

    Supports the nearest-enclosing (parent) lookup and the "direct winner
    children inside an arbitrary segment" queries shared by plan derivation
    and the action-only target partition.  Construction is quadratic; every
    subsequent query is constant- to logarithmic-time in the node count.
    """

    def __init__(self, nodes: list[_WinnerNode]):
        self.nodes = nodes
        self.by_network: dict[Network, _WinnerNode] = {n.network: n for n in nodes}
        self.parent: dict[Network, _WinnerNode | None] = {n.network: None for n in nodes}
        # True trie children (parent is a node) keyed by the parent network.
        self.node_children: dict[Network, list[_WinnerNode]] = {}

        # Interval index for O(log n) containment queries: nodes sorted by
        # their start address, with parallel start / end integer arrays.
        addr_sorted = sorted(nodes, key=lambda n: int(n.network.network_address))
        self.addr_sorted = addr_sorted
        self._starts = [int(n.network.network_address) for n in addr_sorted]
        self._ends = [s + n.network.num_addresses for s, n in zip(self._starts, addr_sorted)]

        # Build the containment parent map in O(n log n) via a radix-style
        # pre-order sweep: network address ascending, then prefix length
        # ascending (parent before child).  The stack holds the chain of
        # not-yet-closed ancestors; each node's parent is the deepest stack
        # entry whose address range still contains it.  Equal-CIDR groups were
        # already reduced to one winner node, so strictly longer prefixes are
        # genuine descendants.
        ordered = sorted(
            nodes,
            key=lambda n: (int(n.network.network_address), n.network.prefixlen),
        )
        stack: list[_WinnerNode] = []
        for node in ordered:
            start = int(node.network.network_address)
            while stack:
                top = stack[-1].network
                top_end = int(top.network_address) + top.num_addresses
                if top_end <= start:
                    stack.pop()
                else:
                    break
            if stack:
                parent = stack[-1]
                self.parent[node.network] = parent
                self.node_children.setdefault(parent.network, []).append(node)
            stack.append(node)
        for kids in self.node_children.values():
            kids.sort(key=lambda n: (int(n.network.network_address), n.network.prefixlen))

        # Memoized direct-child results for segments that are not themselves
        # nodes (subdivision halves): the plan walk visits only O(nodes) such
        # segments, and repeated identical halves share the cached answer.
        self._segment_children_cache: dict[Network, tuple[_WinnerNode, ...]] = {}

    def direct_children(self, segment: Network) -> list[_WinnerNode]:
        """Winner nodes directly under ``segment`` (``segment`` need not be a
        node itself — subdivision halves are supported).

        A node contained in ``segment`` is a direct child exactly when no
        *other* winner node strictly between it and the segment encloses it.
        Node segments read their precomputed trie children directly; non-node
        segments are answered with the interval index (binary search over
        start addresses + nearest enclosing ancestor), never a full node scan.
        """
        if segment in self.by_network:
            return list(self.node_children.get(segment, ()))

        cached = self._segment_children_cache.get(segment)
        if cached is not None:
            return list(cached)

        start = int(segment.network_address)
        end = start + segment.num_addresses
        plen = segment.prefixlen

        # The deepest node whose start is at/before the segment's first
        # address and whose range reaches into it.  CIDRs are non-overlapping
        # unless one contains the other, so at most this single candidate can
        # strictly enclose the whole segment (any earlier-starting enclosing
        # ancestor would itself be that candidate's parent and is reached via
        # the parent chain below).
        enclosing: _WinnerNode | None = None
        idx = bisect.bisect_right(self._starts, start) - 1
        if idx >= 0:
            candidate = self.addr_sorted[idx]
            if (
                self._ends[idx] > start
                and candidate.network.prefixlen < plen
                and segment.subnet_of(candidate.network)
            ):
                enclosing = candidate

        children: dict[Network, _WinnerNode] = {}

        # Nodes that start inside the segment: each is a direct child unless
        # its nearest enclosing winner also sits strictly inside the segment.
        left = bisect.bisect_left(self._starts, start)
        right = bisect.bisect_left(self._starts, end)
        for j in range(left, right):
            node = self.addr_sorted[j]
            if node.network.prefixlen <= plen:
                continue
            up = self.parent.get(node.network)
            if up is not None and up.network.prefixlen > plen and up.network.subnet_of(segment):
                continue
            children[node.network] = node

        # Children of the enclosing node that lie inside the segment (their
        # parent is outside the segment, so they are topmost inside it); they
        # are a subset of the set found above and dedup on the CIDR key.
        if enclosing is not None:
            for kid in self.node_children.get(enclosing.network, ()):
                if kid.network.subnet_of(segment):
                    children[kid.network] = kid

        result = sorted(
            children.values(),
            key=lambda n: (int(n.network.network_address), n.network.prefixlen),
        )
        self._segment_children_cache[segment] = tuple(result)
        return list(result)

    @staticmethod
    def covers_entire_segment(segment: Network, children: list[_WinnerNode]) -> bool:
        """Whether the direct winner children partition ``segment`` fully.

        Direct trie children are pairwise disjoint CIDRs (two CIDRs are
        either disjoint or one contains the other, and the latter cannot hold
        among siblings), so the segment is fully covered exactly when the
        children's address counts sum to the segment's — a constant-time test
        that avoids any ``address_exclude`` geometry.
        """
        return sum(c.network.num_addresses for c in children) == segment.num_addresses

    def enclosing_action(self, target: Network) -> str | None:
        """Action of the longest-prefix winner node strictly enclosing target."""
        best: _WinnerNode | None = None
        for node in self.nodes:
            if (
                node.network.prefixlen < target.prefixlen
                and target.subnet_of(node.network)
            ):
                if best is None or node.network.prefixlen > best.network.prefixlen:
                    best = node
        return None if best is None else best.action


# ---------------------------------------------------------------------------
# Fast action-only partition of one analysis over a target
# ---------------------------------------------------------------------------
def _action_partition(
    trie: _WinnerTrie, target: Network
) -> list[tuple[Network, str | None]]:
    """Partition ``target`` into disjoint CIDR pieces tagged with final action.

    Same winner-trie geometry as the decision map's *actions*, but without the
    quadratic match chains.  Each returned piece is either a winner node's own
    residual surface (its CIDR minus its direct winner children, tagged with
    that winner's action) or uncovered target space tagged ``None``.  Pieces
    are pairwise disjoint and jointly exhaustive over the target.
    """
    pieces: list[tuple[Network, str | None]] = []

    def visit(segment: Network, background: str | None) -> None:
        exact = trie.by_network.get(segment)
        children = trie.direct_children(segment)
        cutters = [c.network for c in children]

        if exact is None:
            # No node on this segment: the gaps between children inherit the
            # background action (None = unmatched); emit the residual gaps.
            if background is not None:
                for gap in (_subtract(segment, cutters) if cutters else [segment]):
                    pieces.append((gap, background))
        else:
            # Node present: its own residual (CIDR minus direct children) is
            # forwarded with the node's action.  A fully shadowed node has no
            # residual and emits nothing.
            for gap in (_subtract(segment, cutters) if cutters else [segment]):
                pieces.append((gap, exact.action))

        for child in children:
            visit(child.network, exact.action if exact is not None else background)

    visit(target, trie.enclosing_action(target))
    pieces.sort(key=lambda p: (int(p[0].network_address), p[0].prefixlen))
    return pieces


class _Emitter:
    """Accumulates the generated deny CIDRs and the un-beatable blockers."""

    def __init__(self, trie: _WinnerTrie):
        self.trie = trie
        self.emitted: list[Network] = []
        self.blockers: list[tuple[Network, _WinnerNode]] = []

    @property
    def nodes(self) -> list[_WinnerNode]:
        return self.trie.nodes

    def emit(self, network: Network) -> None:
        self.emitted.append(network)

    def _direct_children(self, segment: Network) -> list[_WinnerNode]:
        return self.trie.direct_children(segment)

    def recurse_children(self, segment: Network, background: str | None) -> list[_WinnerNode]:
        """Handle the direct winner children of ``segment``.

        ``background`` is the action that addresses of ``segment`` resolve to
        when no child rule matches (the nearest enclosing winner's action, or
        ``None`` when nothing matches there).  Returns the direct winner
        children so the caller can measure its own residual surface.
        """
        children = self._direct_children(segment)
        # The area between the children inherits the segment's background;
        # each child brings its own winner action and is handled on its own.
        for child in children:
            self.walk(child.network, background)
        return children

    def walk(self, segment: Network, background: str | None) -> None:
        exact = self.trie.by_network.get(segment)
        if exact is None:
            # No rule on this exact CIDR: winner children decide their own
            # ranges and are always walked (a permit child under a deny
            # ancestor still needs quarantine rules); the residual surface
            # between them inherits the background action.
            children = self.recurse_children(segment, background)
            if (
                background != QUARANTINE_ACTION
                and not self.trie.covers_entire_segment(segment, children)
            ):
                # Residual is permit (a higher-priority ancestor) or unmatched
                # (none); one deny rule wins the residual by longest prefix
                # without touching the winner children.  A fully shadowed
                # region (children cover the whole CIDR) emits nothing.
                self.emit(segment)
            return

        if exact.action == QUARANTINE_ACTION:
            # The segment itself is already deny; only permit descendants can
            # still need rules, with deny inherited as their background.
            self.recurse_children(segment, QUARANTINE_ACTION)
            return

        # Exact permit winner.
        if exact.priority > QUARANTINE_PRIORITY:
            # A new deny rule at priority 0 wins the exact-prefix tie on the
            # node's residual surface; skip a no-op rule when winner children
            # cover the whole CIDR.
            children = self.recurse_children(segment, QUARANTINE_ACTION)
            if not self.trie.covers_entire_segment(segment, children):
                self.emit(segment)
            return

        # Exact priority-0 permit winner: the new rule cannot win this CIDR
        # (same priority, appended later -> base rule keeps the tie).  If the
        # node covers no address itself (winner children take the whole CIDR)
        # it blocks nothing: just walk its children against the real
        # background.
        children = self._direct_children(segment)
        if self.trie.covers_entire_segment(segment, children):
            for child in children:
                self.walk(child.network, background)
            return

        # The node forwards at least one address of its own.  A single host
        # has no longer prefix to subdivide into.
        if segment.prefixlen >= segment.max_prefixlen:
            self.blockers.append((segment, exact))
            return

        # Subdivide into the two child prefixes: any added rule inside a child
        # has a strictly longer prefix than the priority-0 permit rule and
        # beats it.  A half that coincides with its own exact winner node is
        # handled by the general walk (an existing deny half emits nothing);
        # otherwise the half's winner children are walked against the real
        # enclosing background and the half's own residual (which resolves
        # through the priority-0 permit rule) gets one deny rule at the half.
        for half in segment.subnets(new_prefix=segment.prefixlen + 1):
            if half in self.trie.by_network:
                self.walk(half, background)
            else:
                half_children = self.recurse_children(half, background)
                if not self.trie.covers_entire_segment(half, half_children):
                    self.emit(half)


def _build_plan_networks(
    base: dict[str, Any], target: Network
) -> tuple[list[Network], list[tuple[Network, _WinnerNode]], _WinnerTrie]:
    trie = _WinnerTrie(_winner_nodes(base)[target.version])
    emitter = _Emitter(trie)
    # Addresses of the target resolve through the longest existing winner
    # rule that encloses the whole target (or match nothing when none does).
    emitter.walk(target, trie.enclosing_action(target))
    # Emitted CIDRs are kept exactly as derived: they can legitimately nest
    # around intervening different-action nodes (e.g. a deny /23 plus deny
    # /25s split by a priority-0 permit /24), and geometric collapse would
    # wrongly swallow the deeper rules.
    emitted = _sort_networks(emitter.emitted)
    blockers = sorted(set(emitter.blockers), key=lambda b: (int(b[0].network_address), b[0].prefixlen))
    return emitted, blockers, trie


# ---------------------------------------------------------------------------
# Candidate re-verification
# ---------------------------------------------------------------------------
def _target_scoped_diff(
    base_meta: dict[str, Any],
    base_pieces: list[tuple[Network, str | None]],
    candidate: dict[str, Any],
    candidate_pieces: list[tuple[Network, str | None]],
    target: Network,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Semantic diff restricted to the quarantine target.

    Both partitions are already disjoint and jointly exhaustive over the
    target, so overlaying them with the same geometry as
    :func:`app.diff.diff_analyses` yields exactly the base→candidate changes
    inside the target.  Addresses outside the target cannot change (every
    generated rule is contained in the target), hence they are absent by
    construction.  Returns the diff-shaped payload (this family only) and the
    list of change groups.
    """
    cells = _overlay(list(base_pieces), list(candidate_pieces))
    by_type: dict[tuple[Action, Action], list[Network]] = {}
    for net, a_action, b_action in cells:
        if a_action != b_action:
            by_type.setdefault((a_action, b_action), []).append(net)

    changes: list[dict[str, Any]] = []
    type_counts = {change_type_name(f, t): 0 for f, t in CHANGE_TYPE_ORDER}
    family_total = 0
    for frm, to in CHANGE_TYPE_ORDER:
        nets = by_type.get((frm, to))
        if not nets:
            continue
        collapsed = list(ipaddress.collapse_addresses(_sort_networks(nets)))
        count = sum(n.num_addresses for n in collapsed)
        changes.append(
            {
                "ip_version": target.version,
                "change_type": change_type_name(frm, to),
                "from_action": frm,
                "to_action": to,
                "address_count": count,
                "range_count": len(collapsed),
                "cidrs": [str(n) for n in collapsed],
            }
        )
        type_counts[change_type_name(frm, to)] = count
        family_total += count

    family = {
        "changed_addresses": family_total,
        "change_groups": len(changes),
        "by_type": type_counts,
    }
    summary = {
        "total_changed_addresses": family_total,
        "total_change_groups": len(changes),
        f"ipv{target.version}": family,
    }
    # The other family is untouched; mirror the global diff's shape for it.
    other_version = 6 if target.version == 4 else 4
    other_types = {change_type_name(f, t): 0 for f, t in CHANGE_TYPE_ORDER}
    summary[f"ipv{other_version}"] = {
        "changed_addresses": 0,
        "change_groups": 0,
        "by_type": other_types,
    }

    diff = {
        "base": base_meta,
        "target": {
            "id": candidate.get("id"),
            "ruleset_hash": candidate.get("ruleset_hash"),
            "analyzer_version": candidate.get("analyzer_version"),
        },
        "changes": changes,
        "summary": summary,
    }
    return diff, changes


def build_quarantine_plan(
    base: dict[str, Any],
    target: Network,
    rule_id_prefix: str,
) -> dict[str, Any]:
    """Derive and re-verify the quarantine plan for one saved version.

    On success every generated rule id (``<prefix>1`` ... ``<prefix>N``) must
    be new relative to the base version; colliding ids raise
    :class:`QuarantinePlanError` and no plan is returned.  The candidate is
    rebuilt from the base input, re-analyzed with the existing engine and
    checked through the semantic diff and decision-map machinery: on a
    generatable plan all target blocks end deny, only
    ``permit → deny`` / ``none → deny`` surfaces change, and nothing changes
    outside the target.
    """
    plan_networks, blockers, base_trie = _build_plan_networks(base, target)

    # Already-deny surface inside the target, derived from a fast
    # action-only partition of the base winner trie (no match chains).
    base_pieces = _action_partition(base_trie, target)
    already_deny_nets = list(
        ipaddress.collapse_addresses(
            _sort_networks(net for net, action in base_pieces if action == QUARANTINE_ACTION)
        )
    )

    base_meta = {
        "id": base.get("id"),
        "ruleset_hash": base.get("ruleset_hash"),
        "analyzer_version": base.get("analyzer_version"),
    }
    already_deny = {
        "cidrs": [str(n) for n in already_deny_nets],
        "range_count": len(already_deny_nets),
        "address_count": sum(n.num_addresses for n in already_deny_nets),
    }

    if blockers:
        return {
            "base": base_meta,
            "target_cidr": str(target),
            "ip_version": target.version,
            "rule_id_prefix": rule_id_prefix,
            "generatable": False,
            "rules": None,
            "blocking": [
                {
                    "cidr": str(net),
                    "address_count": net.num_addresses,
                    "rule_id": node.rule_id,
                    "action": node.action,
                    "priority": node.priority,
                }
                for net, node in blockers
            ],
            "converted": None,
            "already_deny": already_deny,
            "candidate": None,
            "candidate_decisions": None,
            "diff": None,
            "summary": {
                "generated_rule_count": 0,
                "blocking_range_count": len(blockers),
                "blocking_address_count": sum(net.num_addresses for net, _ in blockers),
                "converted_address_count": 0,
                "already_deny_address_count": already_deny["address_count"],
                "target_address_count": target.num_addresses,
            },
        }

    # Deterministic ids: canonical CIDR order, sequence starting at 1.
    generated_rules = [
        {
            "rule_id": f"{rule_id_prefix}{i + 1}",
            "cidr": str(net),
            "action": QUARANTINE_ACTION,
            "priority": QUARANTINE_PRIORITY,
            "source_tag": QUARANTINE_SOURCE_TAG,
        }
        for i, net in enumerate(plan_networks)
    ]

    existing_ids = {r["rule_id"] for r in base.get("rules", [])}
    collisions = sorted(
        {r["rule_id"] for r in generated_rules if r["rule_id"] in existing_ids}
    )
    if collisions:
        raise QuarantinePlanError(
            [
                f"generated rule_id {rid!r} already exists in the base version; "
                "choose a different rule_id_prefix"
                for rid in collisions
            ]
        )

    # ---- Rebuild + re-analyze the candidate with the existing engine ------
    candidate_input = rebuild_input_rules(base) + generated_rules
    candidate = analyze_rules(candidate_input)

    # ---- Verify through the fast action partition: target all deny -------
    # The candidate trie is built from the re-analyzed candidate, so this is a
    # genuine recomputation (not an inference from the generated rules).
    candidate_trie = _WinnerTrie(_winner_nodes(candidate)[target.version])
    candidate_pieces = _action_partition(candidate_trie, target)
    non_deny = [(net, action) for net, action in candidate_pieces if action != QUARANTINE_ACTION]
    if non_deny:
        raise QuarantinePlanError(
            [
                "internal: quarantine candidate leaves non-deny blocks inside the target: "
                + ", ".join(str(net) for net, _ in non_deny)
            ]
        )

    # ---- Converted surface = target pieces that were not already deny ----
    # Recomputed by overlaying the base and candidate action partitions over
    # the target only (target-scoped equivalent of the global semantic diff):
    # every non-deny base piece is permit/none -> deny; nothing outside the
    # target can change because every generated rule is contained in it.
    diff, converted_groups = _target_scoped_diff(
        base_meta, base_pieces, candidate, candidate_pieces, target
    )
    for change in converted_groups:
        if change["to_action"] != QUARANTINE_ACTION or change["from_action"] == QUARANTINE_ACTION:
            raise QuarantinePlanError(
                [
                    "internal: quarantine candidate changes "
                    f"{change['change_type']} on {', '.join(change['cidrs'])}; "
                    "only permit/deny and none/deny flips are allowed"
                ]
            )

    converted_nets = [
        ipaddress.ip_network(cidr)
        for change in converted_groups
        for cidr in change["cidrs"]
    ]
    converted_collapsed = list(ipaddress.collapse_addresses(_sort_networks(converted_nets)))
    converted_address_count = sum(n.num_addresses for n in converted_collapsed)
    # Adjacent same-winner residual pieces of one rule are not collapsed here;
    # the public partition collapses adjacent deny pieces (all share deny) so
    # the block list is the minimal CIDR cover of the target.
    deny_pieces = list(
        ipaddress.collapse_addresses(_sort_networks(net for net, _ in candidate_pieces))
    )
    winner_by_network = {n.network: n for n in candidate_trie.nodes}
    candidate_decisions = {
        "target_cidr": str(target),
        "ip_version": target.version,
        "summary": {
            "block_count": len(deny_pieces),
            "deny_addresses": sum(n.num_addresses for n in deny_pieces),
            "permit_addresses": 0,
            "unmatched_addresses": 0,
            "total_addresses": target.num_addresses,
        },
        "blocks": [
            {
                "cidr": str(net),
                "address_count": net.num_addresses,
                "action": QUARANTINE_ACTION,
                "winner_rule_id": winner_by_network.get(net).rule_id
                if net in winner_by_network
                else None,
            }
            for net in deny_pieces
        ],
    }

    return {
        "base": base_meta,
        "target_cidr": str(target),
        "ip_version": target.version,
        "rule_id_prefix": rule_id_prefix,
        "generatable": True,
        "rules": generated_rules,
        "blocking": [],
        "converted": {
            "cidrs": [str(n) for n in converted_collapsed],
            "range_count": len(converted_collapsed),
            "address_count": converted_address_count,
            "by_change": converted_groups,
        },
        "already_deny": already_deny,
        "candidate": candidate,
        "candidate_decisions": candidate_decisions,
        "diff": diff,
        "summary": {
            "generated_rule_count": len(generated_rules),
            "blocking_range_count": 0,
            "blocking_address_count": 0,
            "converted_address_count": converted_address_count,
            "already_deny_address_count": already_deny["address_count"],
            "target_address_count": target.num_addresses,
        },
    }
