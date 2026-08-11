# Known limitations

Stated up front because plan §16.3 requires transparency about system limits, and because
overstating what this repository does would damage exactly the community trust the project depends
on (plan §14, "extractive research perception").

## The lexicon and grammar are provisional

`src/signsync/resources/usl_lexicon.json` and the reordering rules in
`signsync.translation.english_to_sign` are a **structural placeholder**. They were written to
exercise the pipeline, not by USL linguists. Before any output is shown to a user outside the
development team, both must be replaced by entries reviewed against the *Uganda Sign Language
Dictionary* (Kyambogo, 2006) and signed off by the Deaf Advisory Board (plan §6).

Concretely, the current rules encode a simplified topic–comment reordering and a small set of
non-manual markers. Real USL grammar includes spatial referencing, classifier predicates, verb
agreement through space, and role shift — none of which are handled yet.

Three further limits of the translation layer, all visible in its output rather than hidden:

- **One clause per utterance.** A sentence with two clauses is collapsed into a single frame, so
  "I do not understand, where is the hospital?" comes out as one gloss sequence with both a
  negation and a question marker over it. Clause segmentation is not implemented.
- **Modality is not represented.** CAN, MUST and similar have no place in the semantic frame, so
  those words are reported as untranslatable rather than dropped. That is deliberate: dropping
  "must" turns an instruction into a description.
- **Recipients are treated as patients.** `GIVE` is marked as an agreeing verb in the lexicon, but
  the generator does not yet use spatial loci to express who gives to whom.

## There are no trained recognition weights

The repository ships model *architectures* and a training pipeline, not weights. The NumPy
prototype recogniser is a nearest-class-mean classifier: it is enough to demonstrate the end-to-end
loop on a handful of signs, and is not a research result. Any accuracy number produced from
synthetic or sample data is meaningless as evidence about real signing.

## Recognition is isolated-first

Continuous segmentation (`signsync.recognition.segmentation`) uses motion-energy boundary detection
with a duration prior. This is a reasonable baseline, but plan §8.3 is explicit that continuous
signing is the hardest recognition milestone, and a heuristic segmenter is not a solution to
co-articulation.

## Avatar motion is interpolated, not generated

`signsync.motion` implements stages 1 and 2 of plan §8.7 (clip playback, and blended transitions
with inverse kinematics). Stage 3 — a learned motion model for novel sequences — is not implemented.
Motion for any gloss not in the clip library will fall back to fingerspelling or to an explicit
"unknown sign" indication, never to a silently wrong approximation.

## Facial non-manual markers are coarse

The face channel carries a small set of discrete markers (brow raise, brow furrow, head tilt, head
shake, mouth aperture). Real non-manual grammar is continuous and co-articulated. The rig supports
finer control than the generator currently uses.

## Speech input assumes cooperative conditions

The STT adapter is a thin wrapper. Plan §8.5 requires tolerance of Ugandan English accents and
clinic/market background noise; that has not been measured, and no accent-specific evaluation set
exists yet.

## Not a substitute for a human interpreter

Per plan §5 and §16.3, this system is an assistive tool. It must not be used as the interpretation
channel in medical, legal, or safety-critical settings. The API attaches a confidence score to every
translation and the clients surface it; do not remove that.
