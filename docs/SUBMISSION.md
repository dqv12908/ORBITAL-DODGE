# Submission summary

![ORBITAL DODGE thumbnail](images/thumbnail.png)

**One-liner:** ORBITAL DODGE spots dangerous satellite conjunctions two days early. It catches 94.8 % of the encounters that end critical, against 86.6 % for the baseline, while a human makes every call.

**Summary:**
Satellite analysts get hundreds of collision warnings a week, almost all harmless, and must decide about a day before each encounter. ORBITAL DODGE turns that flood into a short, ranked to-do list:
1. Physics recomputes the collision probability. It matches ESA's own values with 98.97 % tier agreement.
2. Rules set the tier.
3. A LightGBM forecaster, chosen from about 540 configurations on our own GPU server, predicts which events will still be critical at the encounter.
4. TypeSafe Jev recommends an action.
5. Claude explains it in plain English, with every number checked.
6. The analyst signs off on a clean decision card.
7. Everything lands in a hash-chained ledger and plays out in a live 3D Earth simulator.

On the organizer's official labels and missions it never saw, it catches 94.8 % of critical events against 86.6 % for the baseline, at 30 % lower expected cost. We report its failures openly.
