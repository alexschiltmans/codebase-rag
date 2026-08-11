# Architecture overview

## Shape

The service is a single process fronted by a load balancer, with three
background concerns run as threads rather than separate deployables: outbound
email delivery, thumbnail generation, and the websocket broadcast hub.

Keeping them in-process is a deliberate trade. It costs isolation, so a memory
leak in thumbnail generation takes the request path down with it. It buys not
having to run and monitor three more things, which at this scale is the larger
cost.

## Storage

One relational database holds everything. Schema changes go through the
forward-only migration ledger; there is no down migration, by design.

Caching is in-process and bounded by entry count rather than by memory, on the
grounds that an entry count is something an operator can reason about while a
memory bound depends on what happens to be cached.

## Authentication

Callers present short-lived bearer tokens carrying an absolute expiry claim.
Tokens are signed rather than stored, so verification needs no database round
trip, and revocation before expiry is deliberately not supported: the token
lifetime is short enough that waiting it out is the revocation mechanism.

## What is not here

There is no message broker, no service mesh, and no separate read replica. Each
was considered and left out because the traffic does not yet justify the
operational surface. The points where one would slot in are the email queue,
the broadcast hub, and the pagination layer respectively.
