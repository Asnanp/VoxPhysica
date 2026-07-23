# VoxPhysica Short-Speaker Collection Protocol v1

## Objective

Collect real, adult, pseudonymous speakers below 160 cm with repeated measured heights and usable speech, while creating a newly sealed external test set. Public HeightCeleb records may support training only because their height labels are internet-derived estimates.

## Pilot quotas

| Role | Measured short speakers | Minimum male speakers |
|---|---:|---:|
| Development support | 120 | 30 |
| Sealed external test | 80 | 15 |
| Total | 200 | 45 |

These are minimum pilot targets, not claims of population representativeness. Recruitment should also cover multiple primary languages, ages, devices, and recording environments. Do not reject non-binary or prefer-not-to-say participants; record the response honestly and evaluate model compatibility separately.

## Ethics gate

Do not recruit or record participants until institutional ethics review, consent wording, data retention, withdrawal, compensation, and access controls are approved. Recruit adults only. Do not scrape private voice recordings or contact people based on inferred height.

## Pseudonymous intake

Use a random code such as `VSP-7F3A91C2`. Keep names and contact details, if needed, in a separate encrypted withdrawal key. The intake CSV rejects direct-identifier columns.

Required consent flags:

- `consent_audio_research=yes`
- `consent_model_training=yes`
- optional `consent_public_release=yes/no`

## Height measurement

1. Use a calibrated stadiometer when possible; otherwise use a fixed vertical surface and rigid headpiece.
2. Participant removes shoes and bulky headwear, stands on a hard level floor, heels together, and looks straight ahead in the Frankfort horizontal plane.
3. Record to the nearest 0.1 cm.
4. Take two independent measurements after repositioning.
5. If they differ by more than 0.5 cm, take a third measurement and document the two closest values.
6. The automated intake rejects a two-measurement difference above 1.0 cm.
7. Record who measured height and the measurement date.

## Audio protocol

Record uncompressed PCM WAV, mono preferred, at 16 kHz or higher. Keep the microphone 20–40 cm from the mouth and avoid automatic voice effects.

Collect at least:

- six prompted sentences containing varied vowels and consonants;
- sustained /a/, /i/, and /u/ vowels, repeated twice;
- counting from 1 to 20;
- 60–90 seconds of natural speech; and
- a second session or second device for at least 25% of participants.

Target at least five quality-controlled clips per person and 8–12 minutes total speech. Keep raw session files; segment copies may be derived later without counting them as new people.

## Split assignment

Assign `collection_role=development` or `collection_role=sealed_test` before model evaluation. Do not move speakers between roles after viewing errors. All clips and sessions from one person stay in one role.

The sealed-test labels must not influence feature selection, model choice, ensemble weights, thresholds, correction rules, or stopping decisions. Report the test only once after the full recipe is frozen.

## Quality controls

The collector verifies:

- unique pseudonymous speaker IDs;
- no speaker overlap with existing train, validation, or historical test;
- readable PCM WAV;
- duration between 2 and 30 seconds per segmented clip;
- sample rate at least 8 kHz;
- non-silent signal;
- clipping fraction at most 1%;
- duplicate audio hashes; and
- at least five valid clips per accepted speaker.

Manual review should additionally check wrong-speaker audio, overlapping speech, synthetic or voice-converted audio, television/music contamination, and label transcription.

## Public HeightCeleb support

The local HeightCeleb/VoxCeleb manifest currently provides 80 speakers below 160 cm with 3,200 available clips. Use them only as noisy train support, preserve attribution, and do not redistribute VoxCeleb audio. The HeightCeleb paper explicitly describes label uncertainty and recommends precise measured data for testing.

## Commands

Audit existing public support:

    python scripts/collect_short_speaker_data.py

Add consented participants:

    python scripts/collect_short_speaker_data.py --consented-csv secure/intake.csv --audio-root secure/audio

Run the development-only support comparison:

    python scripts/evaluate_short_support_dev.py

Neither command turns augmented clips into new people, and the development comparison does not load the historical test.
