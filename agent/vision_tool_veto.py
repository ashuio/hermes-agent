"""TARS-PATCH: unified resolution of "does this route reject images inside tool-result messages".

Three call sites need this answer, and each previously resolved it differently —
name-only at the build-time summary decision and the vision fast-path gate, and
name + base_url-matched custom providers in the executor. The name-only lookups miss
veto profiles under runtime provider canonicalization: the gateway runs a keyed custom
route as ``provider == "custom"`` (model.provider ``custom:bifrost`` canonicalized for
transport) while the operator's veto profile is registered under the qualified id
``custom:bifrost`` (plugins/model-providers/bifrost). Behaviour must not depend on
which spelling a given process happens to use.

Resolution for a route (provider id(s) + base_url + model):

1. ``routed_model_rejects_vision_tool_messages(provider, model)`` for each candidate
   id — covers direct profiles and routing-aggregator targets;
2. every ``custom_providers`` entry whose ``base_url`` matches the active route —
   check its ``custom:<slug>`` profile id (what the operator's plugin registered);
   with a single bare-``custom`` route and no URL match, mirror
   ``resolve_custom_provider``'s self-heal (first valid entry).

Cached per ``(provider, base_url, model)`` — provider identity and endpoint do not
change mid-process except via an explicit model switch, and a switched route arrives
with fresh values (same pattern as ``_AUX_VISION_ROUTE_CACHE`` in computer_use).
Fail-open: any error resolves to False (the pre-patch behaviour), never raises.

Callers:
- ``tools/vision_tools._should_use_native_vision_fast_path`` — route choice (promotion-aware);
- ``agent/vision_message_prep.VisionMessagePrepMixin`` — keep-vs-summarise decision;
- ``agent/tool_executor._commit_tool_result`` — proactive promotion.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# (provider.lower(), base_url_normalized, model) -> route rejects tool-result images?
_VETO_CACHE: Dict[Tuple[str, str, str], bool] = {}


def _normalize_base_url(url: Any) -> str:
    return str(url or "").strip().rstrip("/").casefold()


def _configured_custom_slugs(cfg: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """``(custom:<slug>, normalized_base_url)`` for each configured custom provider, in order."""
    try:
        if cfg is None:
            from hermes_cli.config import load_config_readonly
            cfg = load_config_readonly()
    except Exception as exc:
        logger.debug("vision_tool_veto: config load failed: %s", exc)
        return []
    try:
        from hermes_cli.providers import custom_provider_slug
    except Exception:
        return []
    entries: List[Tuple[str, str]] = []
    for entry in (cfg or {}).get("custom_providers") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        key = str(entry.get("provider_key") or "").strip()
        if not name and not key:
            continue
        entry_url = _normalize_base_url(
            entry.get("base_url") or entry.get("url") or entry.get("api")
        )
        entries.append((custom_provider_slug(name, key), entry_url))
    return entries


def resolve_tool_result_image_veto(
    provider: str,
    model: str,
    *,
    requested_provider: str = "",
    base_url: str = "",
    cfg: Optional[Dict[str, Any]] = None,
) -> bool:
    """True when the active route rejects image parts inside tool-result messages.

    Covers both hard 400s (Hyper Charm ``unsupported input item type``, Command Code,
    Xiaomi MiMo) and silent strips (Bifrost: HTTP 200, image gone). In either case the
    image must be PROMOTED into a user message to reach the model.
    """
    names: List[str] = []
    for candidate in (provider, requested_provider):
        norm = str(candidate or "").strip().lower()
        if norm and norm not in names:
            names.append(norm)
    normalized_url = _normalize_base_url(base_url)
    cache_key = (names[0] if names else "", normalized_url, str(model or "").strip())
    cached = _VETO_CACHE.get(cache_key)
    if cached is not None:
        return cached
    rejects = False
    try:
        from providers import routed_model_rejects_vision_tool_messages

        rejects = any(
            routed_model_rejects_vision_tool_messages(name, model) for name in names
        )
        if not rejects:
            matched: List[str] = []
            entries = _configured_custom_slugs(cfg)
            for slug, entry_url in entries:
                if normalized_url and entry_url and entry_url == normalized_url:
                    matched.append(slug)
            if not matched and "custom" in names and entries:
                # Mirror resolve_custom_provider's self-heal: bare "custom" resolves to
                # the first valid entry when no URL distinguishes it.
                matched = [entries[0][0]]
            rejects = any(
                routed_model_rejects_vision_tool_messages(slug, model) for slug in matched
            )
    except Exception as exc:
        logger.debug("vision_tool_veto: resolution failed: %s", exc)
        rejects = False
    _VETO_CACHE[cache_key] = rejects
    return rejects


def is_custom_endpoint_route(
    provider: str, *, base_url: str = "", cfg: Optional[Dict[str, Any]] = None
) -> bool:
    """True for operator-configured custom endpoints (``custom`` / ``custom:<slug>``).

    These routes are endpoints the operator wired up deliberately, so a tool-image veto
    on their profile means "promote" (the executor moves images into a user message),
    not "fall back to aux text". Third-party profiles keep the upstream conservative
    semantics (#89981: aux fallback / build-time summary).
    """
    name = str(provider or "").strip().lower()
    if name.startswith("custom"):
        return True
    normalized_url = _normalize_base_url(base_url)
    if not normalized_url:
        return False
    try:
        return any(
            entry_url and entry_url == normalized_url
            for _slug, entry_url in _configured_custom_slugs(cfg)
        )
    except Exception:
        return False


__all__ = ["resolve_tool_result_image_veto", "is_custom_endpoint_route"]
