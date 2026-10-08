from __future__ import annotations

import os
import asyncio
import logging

import discord
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

PISTACHIO_TOKEN = os.getenv("DISCORD_TOKEN")    # Pistachio / Tachi account
BLANC_TOKEN     = os.getenv("DISCORD_TOKEN1")   # Blanc account

# The group chat both accounts belong to (the group DM's channel id).
GROUP_CHANNEL_ID = 1311933748438237185

# How often (seconds) each account re-checks that its partner is still present.
POLL_INTERVAL = 5

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("anti_leave")

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

# Each account records its own user id here once it logs in, so the *other*
# account's watcher can look it up at runtime — no hardcoded user ids needed.
account_ids: dict[str, int | None] = {"pistachio": None, "blanc": None}

# Guards against spawning a second watcher if on_ready fires again on reconnect.
_watching: set[str] = set()

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

async def _get_group(client: discord.Client):
    """Return the group channel from cache, falling back to a fetch.

    Returns None if this account can't see the group (e.g. it was itself removed
    — in which case the partner account is the one responsible for re-adding it).
    """
    channel = client.get_channel(GROUP_CHANNEL_ID)
    if channel is not None:
        return channel
    try:
        return await client.fetch_channel(GROUP_CHANNEL_ID)
    except Exception:
        return None


async def ensure_partner_present(
    client: discord.Client, name: str, partner_name: str, partner_id: int
) -> None:
    """Re-add the partner to the group if they are no longer a recipient."""
    channel = await _get_group(client)
    recipients = getattr(channel, "recipients", None)
    if recipients is None:
        # Either we're not in the group, or the channel isn't a group DM.
        return

    if any(u.id == partner_id for u in recipients):
        return  # partner is present — nothing to do

    # Partner is gone → add them back.
    user = client.get_user(partner_id) or await client.fetch_user(partner_id)
    await channel.add_recipients(user)
    log.info("[%s] %s was missing from the group — re-added them.", name, partner_name)


async def watch_partner(client: discord.Client, name: str, partner_name: str) -> None:
    """Poll group membership forever, re-adding the partner whenever they drop."""
    await client.wait_until_ready()
    log.info("[%s] guarding the group — will re-add %s if they leave.", name, partner_name)
    while not client.is_closed():
        partner_id = account_ids.get(partner_name)
        if partner_id is not None:  # wait until the partner account has logged in
            try:
                await ensure_partner_present(client, name, partner_name, partner_id)
            except Exception as exc:
                log.warning("[%s] could not re-add %s: %s", name, partner_name, exc)
        await asyncio.sleep(POLL_INTERVAL)


def build_client(name: str, partner_name: str) -> discord.Client:
    """Create a client for one account that watches for its partner leaving."""
    client = discord.Client()

    @client.event
    async def on_ready() -> None:
        account_ids[name] = client.user.id
        log.info("[%s] logged in as %s (id=%s).", name, client.user, client.user.id)
        if name not in _watching:
            _watching.add(name)
            asyncio.create_task(watch_partner(client, name, partner_name))

    @client.event
    async def on_group_remove(channel, user) -> None:
        # Fires the instant a recipient is removed from a group DM. If it's our
        # group and the partner is the one who left, re-add them right away rather
        # than waiting for the next poll. We add directly (not via the recipient
        # check) since the event itself is proof the partner is gone.
        if getattr(channel, "id", None) != GROUP_CHANNEL_ID:
            return
        partner_id = account_ids.get(partner_name)
        if partner_id is None or user.id != partner_id:
            return
        log.info("[%s] %s was removed from the group — re-adding immediately.", name, partner_name)
        try:
            member = client.get_user(partner_id) or await client.fetch_user(partner_id)
            await channel.add_recipients(member)
        except Exception as exc:
            log.warning("[%s] instant re-add of %s failed: %s", name, partner_name, exc)

    return client

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def main() -> None:
    if not PISTACHIO_TOKEN or not BLANC_TOKEN:
        raise SystemExit(
            "Missing tokens. Set DISCORD_TOKEN (pistachio) and DISCORD_TOKEN1 (blanc) in your .env."
        )

    pistachio = build_client("pistachio", "blanc")
    blanc = build_client("blanc", "pistachio")

    # Run both accounts concurrently in the same process.
    await asyncio.gather(
        pistachio.start(PISTACHIO_TOKEN),
        blanc.start(BLANC_TOKEN),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Shutting down.")
