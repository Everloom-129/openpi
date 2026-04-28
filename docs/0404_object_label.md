# Mask Overlap ratio within 

> dataset: toycubebenchmark/faraz

> wrist camera


Pairwise mask overlap statistics (10518 pairs across all frames):
- mean IoU : 0.0060
- median   : 0.0000
- max      : 0.8070
- std      : 0.0459
- >0.1     : 193 (1.8%)
- >0.3     : 98 (0.9%)
- >0.5     : 19 (0.2%)

Top 15 overlapping pairs (IoU > 0.3):
0.807  'green block'                  ↔ 'pink bowl'  [success/2026-03-25/2026-03-25_12-42-23/perception/00192/perception.h5]

0.721  'red cup'                      ↔ 'bowl'  [failure/2026-03-25/2026-03-25_13-49-07/perception/00072/perception.h5]
> red cup occluded with bowl, SAM2 error

0.651  'pineapple fruit part'         ↔ 'pineapple toy'  [failure/2026-03-25/2026-03-25_13-11-10/perception/00424/perception.h5]
0.638  'red bowl'                     ↔ 'red item'  [failure/2026-03-25/2026-03-25_13-41-48/perception/00344/perception.h5]
0.622  'red bowl'                     ↔ 'bowl'  [failure/2026-03-25/2026-03-25_13-41-48/perception/00360/perception.h5]
0.607  'orange patterned object'      ↔ 'purple bowl'  [success/2026-03-25/2026-03-25_23-14-31/perception/00488/perception.h5]
0.597  'black_power_adapter'          ↔ 'small_black_object'  [failure/2026-03-25/2026-03-25_13-43-24/perception/00328/perception.h5]
0.589  'red block'                    ↔ 'bowl'  [failure/2026-03-25/2026-03-25_13-25-51/perception/00080/perception.h5]
0.589  'pink bowl'                    ↔ 'orange cup'  [failure/2026-03-25/2026-03-25_13-27-45/perception/00072/perception.h5]
0.584  'power_cord_bundle'            ↔ 'power_cord_bundle'  [failure/2026-03-25/2026-03-25_13-43-24/perception/00328/perception.h5]
0.584  'small red cup'                ↔ 'red cup inside white bowl'
[failure/2026-03-25/2026-03-25_13-49-07/perception/00360/perception.h5]
0.574  'large light-orange bowl'      ↔ 'red cup'  [failure/2026-03-25/2026-03-25_13-49-07/perception/00384/perception.h5]
0.566  'red cup'                      ↔ 'white bowl'  [failure/2026-03-25/2026-03-25_13-49-07/perception/00264/perception.h5]
0.557  'red cup inside white bowl'    ↔ 'white bowl'  [failure/2026-03-25/2026-03-25_13-49-07/perception/00360/perception.h5]
0.557  'red cup'                      ↔ 'pink bowl'  [failure/2026-03-25/2026-03-25_13-49-07/perception/00376/perception.h5]

                                                                                                               01:54 PM claude-sonnet-4-6
● The numbers tell a clear story:

  - 98% of pairs have no meaningful overlap — Gemini's segmentations are mostly clean
  - The high-overlap cases (~1%) are mostly container/contained pairs: red cup ↔ bowl, red cup inside white bowl ↔ white bowl — Gemini detected
  both the object and the container as separate items, and SAM2 gave them large overlapping masks
  - A few are straight duplicates: green block ↔ pink bowl at 0.81 IoU is Gemini hallucinating two labels for the same region; pineapple fruit
  part ↔ pineapple toy is the same object named twice

  For the "overlap rule" — the practical approach when building your matching logic would be: if two masks have IoU > 0.3, keep only the one
  whose label better matches the instruction and drop the other. Given that this only affects ~1% of pairs, it's a low-priority edge case, but it
   would eliminate the noisy duplicates that inflate your label vocabulary.
