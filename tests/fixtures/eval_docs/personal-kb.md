# PostgreSQL Backup Policy

The home lab PostgreSQL server takes a full base backup every Sunday at 02:00 UTC and archives write-ahead log segments continuously to object storage. Backups use server-side encryption, and the repository keeps five weekly base backups. A restore drill runs on the first Saturday of each month in an isolated network. The drill is successful only when the restored database passes schema checks and a sampled application query suite.

## Recovery Objectives

The database recovery point objective is fifteen minutes, while the recovery time objective is two hours. WAL archive monitoring pages when no new segment arrives for ten minutes. The restore procedure selects the newest valid base backup before the requested recovery timestamp, replays archived WAL to that timestamp, and records the final transaction identifier in the drill report. Credentials for the backup repository rotate every ninety days.

# Indoor Herb Garden

Basil grows in the south window under a supplemental lamp set to fourteen hours per day. The container uses a two-to-one mixture of potting soil and perlite, with drainage holes kept clear. Water is added when the top two centimeters feel dry rather than on a fixed calendar. A half-strength balanced liquid fertilizer is applied every other week, and flower buds are pinched early to preserve leaf production.

## Pest Response

Fungus gnats are controlled by allowing the surface layer to dry, placing yellow sticky cards beside the pots, and treating the next two waterings with beneficial nematodes. Neem oil is not used on herbs intended for immediate harvest. If aphids appear, the first response is a strong rinse followed by inspection of leaf undersides every two days. An infested pot stays separated from the other plants for at least one week.
