# Deployment runbook

## Prerequisites

Before a release goes out, confirm the migration ledger is empty of pending
entries on the staging database and that the previous release has been running
for at least one full traffic cycle.

## Rolling out

Deployment is blue-green. The new version is brought up alongside the running
one, health checked, and only then given traffic. Nothing is torn down until
the new fleet has served real requests for ten minutes.

1. Build the release image and push it to the registry.
2. Bring up the green fleet with the release tag.
3. Wait for every instance to report healthy on its readiness endpoint.
4. Shift traffic in three steps: ten percent, fifty percent, then all of it.
5. Hold at each step long enough to see error rates for that slice.

## Rolling back

A rollback is a traffic shift back to the blue fleet, not a redeploy. The blue
fleet stays up for one hour after a release precisely so that a rollback is a
routing change measured in seconds rather than an image build measured in
minutes.

Schema migrations are the exception. A migration that dropped a column cannot
be undone by shifting traffic, which is why migrations are forward-only and
additive: a release that needs a column gone ships the code that stops using it
first, and the drop follows in a later release.

## Post-release checks

Watch the error rate, the p99 latency, and the queue depth for outbound email
for thirty minutes. A rise in queue depth without a rise in error rate usually
means the transport is throttling rather than failing.
