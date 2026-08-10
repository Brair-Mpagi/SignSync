# Bidirectional Ugandan Sign Language ↔ Spoken English Translation System
### A Complete Project Plan

---

## 0. Executive Summary

This project builds a real-time, bidirectional translation system between **Ugandan Sign Language (USL)** and spoken English, combining computer vision, temporal deep learning, natural language processing, speech technology, and 3D motion generation. It progresses deliberately from isolated-sign recognition to continuous, sentence-level, conversational translation — and from pre-recorded sign playback to natural, avatar-driven sign generation.

The plan is scoped for **Uganda specifically**, not sign language in the abstract. USL has been constitutionally recognised since 1995 — Uganda was the second country in the world to do so — and is used by an estimated 160,000 native signers. It has an active institutional ecosystem (Uganda National Association of the Deaf, Uganda National Association of Sign Language Interpreters, and a dedicated academic department at Kyambogo University) that this project should engage from day one rather than treat as an afterthought.

The central design principle carried over — and strengthened — from the original concept remains:

> **Do not build a gesture classifier and call it a translator. Build a language system that happens to use vision as one of its inputs and motion as one of its outputs.**

What this revision adds: grounding in existing USL-specific research and datasets, a realistic team/budget/timeline structure, a community-engagement and consent framework aligned with Ugandan law, a sustainability and deployment strategy suited to real connectivity/hardware constraints, and a risk register with concrete mitigations rather than aspirations.

---

## 1. Problem Statement

Communication between Deaf and hearing people breaks down when the two parties do not share a language. In Uganda this gap is not hypothetical — it is documented in daily life: Deaf patients report having to lip-read and write with health workers rather than sign, because interpreters are scarce outside major institutions and formal settings. Uganda has an estimated 100–200 trained interpreters nationally, which is far too few to cover healthcare, education, courts, local government, and everyday commerce for a signing population two orders of magnitude larger.

Existing sign-language recognition research — including prior USL-specific work (see Section 3) — has generally focused on:

- isolated sign / alphabet classification;
- small, fixed vocabularies;
- single-signer or lab-controlled data;
- one-directional recognition only (sign → text), with no generation of sign output.

This does not solve the communication problem end-to-end, because natural signing is continuous, contextual, and expressed through hands, arms, body posture, head movement, and facial expression together — and because true communication requires the *reverse* direction too: a hearing person who does not sign needs a way to produce USL back.

A practical system must therefore treat this as a **language translation problem with two visual/audio surfaces**, not a gesture-recognition problem with a text label attached.

---

## 2. Context: Ugandan Sign Language and Why It Is the Right First Target

- **Legal status:** USL has been recognised in Uganda's Constitution since 1995. The Uganda Communications Act (2013) requires broadcast stations to provide sign-language interpretation for information programming, which signals an existing policy environment favourable to accessibility technology.
- **Community size and structure:** Estimated at roughly 160,000 native signers. The Uganda National Association of the Deaf (UNAD), founded in 1973, has branches in over 110 districts and is the recognised representative body for the Deaf community; the Uganda National Association of Sign Language Interpreters (UNASLI), formed by UNAD in partnership with Kyambogo University, represents professional interpreters.
- **Existing linguistic resources:** UNAD and Kyambogo University's Sign Language Dictionary Research Project have already produced print references — the *Manual of Ugandan Signs* (UNAD, 1998/1999) and the *Uganda Sign Language Dictionary* (Kyambogo University, 2006, ~440 pages, compiled with Wallin, Lule, Lutalo-Kiingi and Busingye). These are valuable for building an initial sign vocabulary and gloss set, though they are not video corpora and cannot substitute for one.
- **Academic infrastructure:** Kyambogo University's Faculty of Special Needs and Rehabilitation runs a dedicated Department of Hearing Impairment and Sign Language Interpretation Studies, including a new (2023) MSc in Ugandan Sign Language Translation and Interpreting — a natural academic partner and source of linguistic expertise, USL-fluent evaluators, and potentially students for dataset collection.
- **Prior technical work:** A 2024–2025 Makerere University project ("Real-Time Translation of Ugandan Sign Language to Speech," Luwaga Micheal et al., published at the 5th Congress on Intelligent Systems, Springer LNNS vol. 1278) used MediaPipe-based pose and hand-landmark extraction with CNN backbones (ResNet50/VGG19) and transformer blocks to classify a set of USL gestures, reporting very high accuracy on its own held-out test data. This confirms the landmark-based approach is viable for USL specifically and is a useful architectural reference — but it is an **isolated-gesture classifier**, not a continuous, sentence-level, bidirectional translator with sign generation. That gap is precisely this project's contribution, and the two teams (or their published methodology) are worth contacting as a starting point rather than duplicating from zero.
- **Comparable regional work:** Kenyan Sign Language and Nigerian Sign Language projects offer transferable lessons on dataset design — in particular, the use of gloss annotation and HamNoSys-style phonetic notation alongside video, which this project should adopt for its own annotation schema (Section 9.3).

This context matters practically: it changes the dataset strategy (there is a print vocabulary to seed from, but no reusable video corpus), the partnership strategy (UNAD, UNASLI, and Kyambogo are prerequisites, not optional stakeholders), and the legal strategy (Uganda's Data Protection and Privacy Act, 2019, governs everything the project records — see Section 16).

---

## 3. Related Work Summary

| Project | Direction | Scope | Relevance |
|---|---|---|---|
| Makerere USL-to-Speech (2024–25) | USL → Speech | Isolated gestures, landmark + CNN/transformer classifier | Closest prior art; validates MediaPipe + transformer approach for USL; no continuous signing, no reverse direction |
| Kenyan Sign Language pose/word datasets | KSL → text/avatar | Static + dynamic gesture dataset, HamNoSys gloss annotation | Template for dataset schema and annotation methodology |
| Nigerian Sign Language sign-to-speech | NSL → Speech | Small dataset, YOLO-based detector, real-time deployment | Shows a low-resource, low-compute deployment pattern worth reusing |
| South African speech-to-sign scoping review | Speech → Sign | Literature review, no working system reported | Confirms speech→sign generation is the least mature direction across the region — expect this project's Phase 6–8 to be the hardest and most novel part |

**Implication:** the sign → English direction has regional precedent to build on; the English → sign direction (translation, motion generation, avatar) is comparatively unexplored for African sign languages and is where this project can make its most original contribution — but should also be budgeted as the highest-risk, highest-effort component.

---

## 4. Vision, Goal, and Objectives

### 4.1 Long-Term Vision

An AI communication bridge that behaves like a language interpreter, not a gesture-recognition demo:

```
                 HUMAN COMMUNICATION
                         │
             ┌───────────┴───────────┐
             │                       │
     Ugandan Sign Language        English
             │                       │
             ↓                       ↓
      Computer Vision          Speech Recognition
             │                       │
             └───────────┬───────────┘
                         ↓
                 AI LANGUAGE CORE
                         │
              ┌──────────┴──────────┐
              ↓                     ↓
           English                USL
              │                     │
              ↓                     ↓
             TTS               Motion Generation
              │                     │
              ↓                     ↓
           🔊 Speech             🤟 Avatar
```

### 4.2 Primary Goal

Deliver a real-time, bidirectional translation platform between USL and spoken English, progressing from isolated-sign recognition to continuous sentence-level translation and natural sign generation — validated by fluent Deaf signers, not only by automatic metrics.

### 4.3 Specific, Measurable Objectives

| # | Objective | Target by end of project |
|---|---|---|
| O1 | Visual tracking pipeline (hands, pose, face) | ≥25 FPS on a mid-range laptop CPU, no GPU required |
| O2 | Landmark-based sign representation | Structured (x,y,z) feature vectors, signer-normalised |
| O3 | Isolated sign recognition | ≥90% top-1 accuracy on a 300–500 sign vocabulary, signer-independent split |
| O4 | Continuous sign recognition | Functional segmentation + recognition on ≥50 test sentences from held-out signers |
| O5 | USL → English translation | Fluent, meaning-preserving English judged acceptable by ≥80% of Deaf/interpreter evaluators |
| O6 | English speech output | Natural TTS output, <1s added latency |
| O7 | English speech recognition | Functional on conversational English with typical background noise |
| O8 | English → USL translation | Grammatically appropriate USL gloss sequences, not word-for-word substitution |
| O9 | Sign motion generation | Smooth, continuous avatar motion rated "understandable" by ≥70% of Deaf evaluators |
| O10 | 3D avatar rendering | Runs in-browser via WebGL/Three.js on common hardware |
| O11 | Real-time integration | End-to-end round-trip latency low enough for turn-taking conversation (target <2s per exchange) |

---

## 5. Scope

**In scope (this project):**
- Ugandan Sign Language only, as the first target language.
- A defined, growing vocabulary (50 → 500+ signs) plus continuous-sentence recognition on a constrained domain first (e.g., greetings, health, education, everyday needs) before open-domain conversation.
- Both directions: USL → English (speech) and English (speech) → USL (avatar).
- A locally deployable system (works with degraded or no internet connectivity for the core recognition/generation loop).

**Out of scope for v1 (explicitly deferred):**
- Other sign languages (architecture should allow adding them later — see Section 5.1 — but v1 does not attempt multi-language support).
- Fully open-domain conversation (v1 targets defined domains; open-domain is a post-MVP research goal).
- Replacing professional human interpreters in legal, medical-emergency, or safety-critical settings — the system is positioned as an assistive tool, never a certified interpretation substitute (see Section 17).

### 5.1 Language Extensibility (Future)

```
                 Translation Engine
                        │
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
       USL             ASL             KSL
        │               │               │
        └───────────────┼───────────────┘
                        ↓
                     English
```

---

## 6. Stakeholders and Community Engagement Plan

Sign-language technology built without the Deaf community routinely fails on real signers even when it performs well on its own training data. This project treats community engagement as a workstream with its own budget line and milestones, not a courtesy consultation.

| Stakeholder | Role in the project |
|---|---|
| Uganda National Association of the Deaf (UNAD) | Community liaison, access to signers for data collection, cultural/linguistic review, dissemination |
| Uganda National Association of Sign Language Interpreters (UNASLI) | Interpreter panel for gloss annotation, evaluation, and gold-standard translation checking |
| Kyambogo University — Dept. of Hearing Impairment & Sign Language Interpretation Studies | Academic partnership, linguistic advisory board, access to the existing USL dictionary, student researchers/evaluators |
| District Associations of the Deaf (via UNAD's 110+ branches) | Regional signer diversity for dataset collection (age, dialectal variation) |
| Deaf schools (e.g., institutions historically linked to UNAD/UNISE) | Recruitment of signers across age groups, education-domain vocabulary |
| National Union of Disabled Persons of Uganda (NUDIPU) | Broader disability-rights alignment, advocacy, funding introductions |
| Project's own Deaf Advisory Board (to be formed in Phase 0) | Ongoing sign-off on translation quality, avatar naturalness, and product decisions — should include Deaf individuals, not only hearing interpreters |

**Principle:** "Nothing about us without us." Every phase that touches sign quality (dataset design, annotation, evaluation, avatar naturalness) must include Deaf reviewers, not only interpreters or engineers.

---

## 7. System Architecture

### 7.1 High-Level Architecture

```
                    ┌─────────────────────┐
                    │       CLIENT        │
                    │ Web / Desktop / App │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ↓                             ↓
          Camera Input                 Microphone Input
                │                             │
                ↓                             ↓
       Computer Vision                 Speech Recognition
                │                             │
                ↓                             ↓
       Landmark Extraction                  English
                │                             │
                ↓                             │
       Temporal Sign Encoder                 │
                │                             │
                ↓                             │
       Sign Recognition                      │
                │                             │
                └──────────────┬──────────────┘
                               ↓
                       Semantic Representation
                               │
                 ┌─────────────┴─────────────┐
                 ↓                           ↓
             English                        USL
                 │                           │
                 ↓                           ↓
                TTS                    Motion Generator
                 │                           │
                 ↓                           ↓
              Speech                    3D Avatar
```

### 7.2 Design Principle: Intermediate Representation

The system deliberately avoids `SIGN → English word` mapping. Sign-language grammar (topic-comment structure, non-manual markers, spatial referencing) differs substantially from English grammar, so a word-for-word approach produces broken output in both directions.

```
Video → Visual Features → Sign/Gloss Units → Semantic Representation → English
English → Language Understanding → Semantic Representation → USL Gloss Sequence → Motion
```

---

## 8. Core Technical Components

### 8.1 Computer Vision Layer

Extracts structured landmarks rather than training directly on raw video.

- **Tools:** MediaPipe Holistic (or equivalent) for hand (21 landmarks × 2 hands), body pose (33 landmarks), and face (subset of the 468-point face mesh — only the landmarks relevant to non-manual grammatical markers: eyebrows, mouth shape, head tilt); OpenCV for capture/preprocessing; ONNX Runtime for optimised inference.
- **Representation:** each frame becomes a structured `(x, y, z)` landmark vector per tracked point, normalised for signer position/scale so the model generalises across body sizes and camera distances.
- **Why landmarks, not raw video:** far smaller input dimensionality → feasible to train on a modest dataset, faster inference, better generalisation across skin tones/lighting/clothing than raw-pixel models, and it directly enables the signer-independent evaluation goal in Section 15.

### 8.2 Temporal Sign Recognition

Sign language is inherently temporal; single frames are ambiguous.

```
Frame 1 … Frame N → Temporal Model → Sign
```

Staged model progression (deliberately incremental — do not start with the most complex model):

1. **Baseline:** LSTM/GRU over landmark sequences.
2. **Improved:** Temporal Convolutional Network.
3. **Advanced:** Transformer-based temporal encoder (matches the approach validated by the Makerere USL project).
4. **Multimodal (later):** fuse hand + body + face landmark streams with optional RGB features.

### 8.3 Continuous Sign Recognition

Moves from `one clip → one sign` to `continuous video → sign sequence → sentence`. This is the hardest recognition milestone and should be resourced accordingly:

- sign boundary / co-articulation detection;
- handling repetition, pauses, and varying signing speed;
- distinguishing near-identical signs in context;
- signer-independent generalisation (train/test split by signer identity, never by clip).

### 8.4 USL → English Translation

```
Sign sequence → Semantic encoder → Translation model → "Where is the hospital?"
```

Architecture: sign encoder → transformer → English decoder. Evaluated with BLEU/ROUGE **and** mandatory human evaluation by Deaf signers and UNASLI interpreters — automatic metrics alone are not sufficient for sign-language translation quality (Section 15).

### 8.5 English Speech Recognition

Standard STT, modularised so different models/engines can be swapped without redesigning the pipeline. Must tolerate conversational speech, typical background noise, and a range of English accents encountered in Uganda.

### 8.6 English → USL Translation

```
English → Language understanding → USL semantic representation → Sign/gloss sequence
```

Learns USL structure (topic-comment ordering, classifiers, spatial grammar) rather than substituting a sign for every English word — the same principle as 7.2, applied in reverse.

### 8.7 Sign Motion Generation

The most technically demanding component. Naive keyframe-to-keyframe interpolation produces robotic motion that Deaf users reject regardless of "correctness."

Staged approach:
1. **Initial:** play back recorded motion-capture or annotated video-derived sign clips.
2. **Intermediate:** blend/interpolate between recorded clips using inverse kinematics for smooth transitions.
3. **Advanced (research-stage, post-MVP):** train a neural motion-generation model (transformer or diffusion-based) for novel sign sequences not directly in the recorded set.

Must jointly model hand shape/orientation, wrist rotation, arm trajectory, body posture, head movement, facial expression, and timing — non-manual markers are not decoration, they carry grammatical meaning (negation, questions, conditionals) and omitting them changes sentence meaning.

### 8.8 3D Avatar

```
English → Sign Translation → Sign Tokens → Motion Generator → Skeleton Motion → IK → 3D Avatar → Rendered Signing
```

Renders via Three.js/WebGL for browser accessibility and low deployment friction; articulated rig needs full hand, arm, upper-body, head, and facial controls — a rig without expressive hands and face cannot render intelligible USL regardless of how good the motion model is.

---

## 9. Dataset Strategy

### 9.1 What Already Exists (and What Doesn't)

| Resource | Type | Usable as |
|---|---|---|
| *Manual of Ugandan Signs* (UNAD, 1998/99) | Print reference | Seed vocabulary list, gloss definitions |
| *Uganda Sign Language Dictionary* (Kyambogo, 2006) | Print reference, ~440 pages | Larger seed vocabulary, sign descriptions |
| Makerere USL-to-Speech project dataset (2024/25) | Video + gesture labels | Possible partnership/reference for isolated-sign data (requires direct contact and permission — not publicly assumed available) |
| Continuous, conversational USL video corpus | — | **Does not appear to exist publicly.** Must be built by this project. |

**Conclusion:** there is no shortcut around original data collection for continuous/conversational USL. Budget and timeline must reflect that this is a data-engineering project as much as a modelling project.

### 9.2 Dataset Phases

| Phase | Vocabulary/content | Purpose |
|---|---|---|
| V1 | 50–100 common signs (greetings, needs, health, numbers) | Prove the CV pipeline; baseline recognition |
| V2 | 300–500 signs | Meaningful vocabulary; simple sentence construction |
| V3 | Continuous sentences in constrained domains | Sequence recognition and translation |
| V4 | Conversational, semi-scripted dialogues | Real-world robustness, non-manual markers, natural pacing |

Exact vocabulary sizes should be finalised in Phase 0 jointly with the Deaf Advisory Board and Kyambogo linguists, prioritising domains with real communication urgency (health, education, government services) over generic dictionary coverage.

### 9.3 Collection and Annotation Protocol

- **Signer diversity:** recruit across age, gender, region/district, signing speed, and background (school-taught vs. home-sign-influenced), using UNAD's district network rather than a single convenient location — this directly targets Risk 2 (poor generalisation) in Section 14.
- **Recording conditions varied deliberately:** camera distance/angle, lighting, background, clothing — because a model trained on one studio setup will fail on a phone camera in a clinic.
- **Annotation schema:** video + gloss label + English translation + start/end timestamps + non-manual marker notes + signer metadata (with consent — see Section 16), following the gloss + phonetic-notation pattern used in the Kenyan Sign Language dataset work (HamNoSys-style annotation) so the corpus is reusable by other researchers.
- **Every collection session is compensated and consented** — signers are domain experts contributing linguistic labour, not free crowd-sourced clicks.

### 9.4 Augmentation

Horizontal translation, scaling, temporal stretching/compression, small rotations, landmark noise, frame dropping, lighting/background variation — applied conservatively, since aggressive temporal or spatial distortion can change a sign's meaning (e.g., direction and orientation are often grammatically significant in sign languages).

---

## 10. Recommended Technology Stack

| Layer | Technology |
|---|---|
| AI/ML | Python, PyTorch, NumPy, scikit-learn |
| Computer vision | OpenCV, MediaPipe (or equivalent landmark tracker) |
| NLP | Hugging Face Transformers, Sentence-Transformers, tokenizers |
| Speech | Modular STT (e.g., Whisper-class model) + TTS engine |
| Backend | Python, FastAPI |
| Frontend | React, TypeScript, WebRTC (camera/mic streaming) |
| 3D rendering | Three.js (browser-first); Blender for asset/motion authoring |
| Deployment | Docker, ONNX Runtime, optional GPU inference, local-first inference where feasible |

---

## 11. Team and Roles

A realistic minimum team, sized for a phased build rather than all-at-once development:

| Role | Responsibility | When needed |
|---|---|---|
| Project Lead / PM | Roadmap, stakeholder coordination, community partnerships | Throughout |
| Computer Vision / ML Engineer(s) | Landmark pipeline, sign recognition models | Phase 1 onward |
| NLP Engineer | Translation models (both directions) | Phase 4 onward |
| Speech Engineer | STT/TTS integration | Phase 5–6 |
| Motion/3D Engineer or Animator | Motion generation, IK, avatar rigging | Phase 7–8 |
| Frontend/Full-stack Developer | Client app, real-time pipeline integration | Throughout, ramping up Phase 9 |
| USL Linguist / Consultant (Kyambogo or UNASLI) | Gloss schema, grammar review, translation QA | Throughout — not optional |
| Deaf Community Liaison | Recruitment, consent process, Advisory Board coordination | Throughout, critical in Phase 0 and data phases |
| Data Annotators (contracted, ideally Deaf or fluent signers) | Gloss annotation, quality control | Data collection phases |
| QA / Evaluation Coordinator | Runs human evaluation rounds, tracks metrics | Phases 2, 3, 4, 7, 9 |

For a small team or student project, several of these roles can be combined, but the **USL linguist and Deaf community liaison roles should never be dropped** — they are what prevents this from becoming "a gesture classifier called a translator."

---

## 12. Development Phases and Timeline

Durations are effort estimates for a small, focused team; adjust for team size and funding.

| Phase | Deliverable | Duration |
|---|---|---|
| 0 — Research & Partnerships | Research report, system spec, signed partnerships with UNAD/UNASLI/Kyambogo, Deaf Advisory Board formed | 3–4 weeks |
| 1 — Computer Vision Prototype | Real-time hand/pose/face landmark tracking demo | 2–4 weeks |
| 2 — Isolated Sign Recognition | Trained model + dataset pipeline (V1→V2 vocabulary) + real-time inference demo | 6–10 weeks |
| 3 — Continuous Sign Recognition | Segmentation + sequence recognition prototype | 8–12 weeks |
| 4 — Sign → English Translation | USL sequence → English sentence model | 6–10 weeks |
| 5 — English → Speech | TTS integration; complete Sign→English→Speech pipeline | 1–2 weeks |
| 6 — English → Sign Representation | English → USL gloss sequence model | 6–10 weeks |
| 7 — Sign Motion Generation | Smooth motion from recorded + interpolated clips | 8–14 weeks |
| 8 — 3D Avatar | Rigged avatar performing generated sign sequences | 6–10 weeks |
| 9 — Full Bidirectional Integration | End-to-end app: both directions, real-time | 6–10 weeks |

**Approximate total: 12–18 months** for a small dedicated team to reach a genuinely bidirectional, continuous-sentence MVP (not the full open-domain vision — see Section 18 for what "done" means at MVP scale).

```
Phase:      0   1   2   3   4   5   6   7   8   9
Focus:    Setup CV  Iso Cont S2E TTS E2S Motion Avatar Integrate
```

---

## 13. Budget and Resource Estimate (Indicative)

Figures are planning-level ranges, not quotes — intended to size fundraising conversations, not to be treated as fixed. Costs assume Uganda-based data collection with international cloud/compute costs.

| Category | Notes | Rough range (USD) |
|---|---|---|
| Personnel (team in Section 11, 12–18 months) | Largest cost; varies hugely with team composition and whether roles are volunteer/student/paid | $20,000 – $150,000+ |
| Signer/annotator compensation | Fair pay for Deaf signers and interpreters contributing data and review — non-negotiable | $3,000 – $10,000 |
| Compute (training + inference) | Cloud GPU credits or local GPU workstation | $2,000 – $15,000 |
| Recording equipment | Cameras, tripods, lighting, portable kits for district visits | $1,500 – $5,000 |
| 3D/animation assets and tools | Avatar rig, motion capture or reference clips, software licenses | $1,000 – $8,000 |
| Community engagement | Advisory Board stipends, workshops, travel to district associations | $2,000 – $6,000 |
| Legal/compliance | NITA-U/PDPO registration, data protection review, consent process design | $500 – $2,000 |
| Contingency | Standard 15–20% buffer | — |

**Funding avenues worth exploring:** disability-inclusion and assistive-technology grant programs (e.g., from international development agencies active in Uganda), World Federation of the Deaf partnerships, university research grants (Makerere, Kyambogo), corporate accessibility/CSR programs, and NUDIPU's funding network.

---

## 14. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Insufficient/low-diversity dataset | High | Very High | Dedicated data pipeline from Phase 0; leverage UNAD's district network; budget explicit annotator/signer compensation |
| Poor generalisation to unseen signers | High | High | Signer-independent train/test splits from the start; deliberately diverse recruitment (Section 9.3) |
| Continuous signing is much harder than isolated signs | High | High | Treat Phase 3 as a major milestone with its own budget/time, not a quick extension of Phase 2 |
| Robotic, unintelligible avatar motion | Medium–High | High | Start from real recorded motion; every motion iteration reviewed by Deaf evaluators before advancing |
| Latency too high for real-time use | Medium | Medium | Landmark-based (not raw-video) models; ONNX/quantization; local inference; measure FPS/latency continuously, don't assume |
| Literal/word-for-word translation errors in either direction | High | High | Semantic intermediate representation (Section 7.2), not dictionary substitution; human evaluation gate before any release |
| Community trust / extractive research perception | Medium | Very High | Deaf Advisory Board with real decision authority from Phase 0; fair compensation; "nothing about us without us" applied structurally, not rhetorically |
| Data protection non-compliance | Medium | High (legal + reputational) | Formal registration with Uganda's Personal Data Protection Office; explicit consent protocol (Section 16); minimal retention |
| Scope creep toward "support every sign language" or full open-domain conversation in v1 | Medium | Medium | Explicit scope boundaries (Section 5); MVP definition (Section 18) reviewed at each phase gate |
| Key personnel dependency (e.g., only one CV engineer, only one linguist) | Medium | Medium | Document architecture and models; cross-train where team size allows; maintain relationship with Kyambogo as an institutional (not individual) partnership |

---

## 15. Evaluation Framework

Accuracy alone is not sufficient evidence of success for a translation system — this is stated explicitly to prevent the project from declaring victory on a benchmark that doesn't reflect real communication.

| Area | Metrics |
|---|---|
| Isolated sign recognition | Accuracy, precision, recall, F1, confusion matrix (per-sign, to catch systematically confused sign pairs) |
| Continuous recognition | Sequence accuracy, sign error rate, temporal alignment quality |
| USL → English translation | BLEU, semantic similarity, **human evaluation by Deaf signers/UNASLI interpreters** on meaning preservation and grammaticality |
| English → USL translation | Human evaluation on grammatical appropriateness (not word-for-word check), meaning preservation |
| Speech recognition | Word error rate, latency |
| Sign motion generation | Smoothness, trajectory error, temporal consistency, **human realism/intelligibility ratings from Deaf evaluators** |
| Overall system | End-to-end latency, successful-communication rate in real conversational tests, robustness across lighting/camera/network conditions, user satisfaction |

**Human evaluation is mandatory, not optional**, drawn from: fluent Deaf signers, UNASLI-certified interpreters, and hearing users unfamiliar with the system (to test genuine usability, not just insider approval).

---

## 16. Ethical, Legal, and Regulatory Considerations

### 16.1 Data Protection Compliance

The project will collect video, and potentially audio and biometric-style landmark data, of identifiable individuals — this falls squarely under Uganda's **Data Protection and Privacy Act, 2019** (and its 2021 Regulations), enforced by the Personal Data Protection Office (PDPO) under NITA-U. Concretely, this means:

- **Registration:** the project (as data controller) should register with the PDPO, as required for any person, institution, or public body collecting/processing personal data.
- **Consent as a central principle:** explicit, informed consent must be obtained from every recorded participant, with distinct consent for participants under 18 (parent/guardian consent required).
- **Purpose limitation and retention limits:** collected video/data should be used only for the stated research/product purpose and not retained longer than necessary.
- **Data subject rights:** participants must be able to understand what is collected, why, and be able to request deletion.
- **Cross-border transfer safeguards:** if any data or model training touches cloud infrastructure outside Uganda, transfer safeguards equivalent to the Act's protections are required.
- **Breach notification:** a plan for notifying affected individuals and the PDPO in the event of a data breach should exist before data collection begins, not after.

A formal Data Protection Impact Assessment should be one of Phase 0's deliverables, not something added later.

### 16.2 Consent Design for a Deaf Community

Standard written consent forms are not sufficient by themselves. Consent processes must be delivered in USL (via a fluent signer, not just an interpreter reading English text aloud), and should explain, in accessible terms: what is recorded, how it will be used, who can access it, how long it is kept, and how to withdraw consent later.

### 16.3 Broader Ethical Commitments

- Position the system explicitly as an **assistive communication tool**, never a replacement for the Deaf community's own language, culture, or for professional certified interpreters in medical, legal, or safety-critical settings.
- Be transparent about system limitations in-product (e.g., confidence indicators, "translation may be imperfect" framing) rather than implying flawless translation.
- Ensure dataset ownership and any downstream commercial value are addressed with UNAD/participants up front, not after the fact.
- Actively represent signing-style diversity (age, region, education background) rather than defaulting to whichever signers were easiest to reach.

---

## 17. Deployment and Sustainability Strategy

- **Local-first inference:** given real connectivity variability, the recognition and generation pipeline should run acceptably on-device or on local/edge infrastructure, with cloud use as an enhancement, not a hard dependency.
- **Low hardware bar:** landmark-based models (Section 8.1) keep compute requirements modest; the 3D avatar should run in-browser via WebGL so it works on common laptops/phones without a dedicated GPU.
- **Institutional deployment path:** pilot first in a small number of high-value settings identified with UNAD (e.g., a health clinic, a school for the Deaf, a government service desk) rather than launching broadly and untested.
- **Long-term custodianship:** define early who maintains the dataset, models, and avatar assets after initial funding/team involvement ends — ideally a Ugandan institution (Kyambogo University and/or UNAD) as a named long-term custodian, so the project doesn't become dependent on any single individual or funding cycle.
- **Open resources where appropriate:** consider releasing the annotated USL dataset (with participant consent already covering this use) as a research resource, following the precedent of regional efforts like the Kenyan Sign Language dataset work — this benefits the wider low-resource sign-language research community and Uganda's academic profile in the field.

---

## 18. Minimum Viable Product (MVP) Definition

### 18.1 First MVP

```
Camera → Hand+Pose+Face Tracking → Temporal Sign Recognition
   → 50–100 Signs → English Text → Text-to-Speech
```

Example: user signs **"HELP"** → system outputs the English text "Help." and speaks it aloud.

### 18.2 Advanced MVP (post-first-MVP milestone)

```
Continuous signing → Multiple signs → English sentence → Natural speech
```

Example: continuous signing → "I need help at the hospital." → spoken aloud.

### 18.3 Full System — Three Modes

**Mode A — Sign to Speech:** Camera → AI → English → Speech
**Mode B — Speech to Sign:** Speech → AI → Sign sequence → 3D Avatar
**Mode C — Conversation Mode:** two users, each seeing/hearing the other's translated output in real time — the ultimate product vision, and the point at which the system genuinely functions as a communication bridge rather than a demo.

---

## 19. Success Criteria

The project is successful when it demonstrably achieves:

1. Reliable real-time sign detection.
2. Recognition that generalises to signers not seen during training.
3. Continuous sign-sequence recognition on constrained-domain sentences.
4. Meaning-preserving USL → English translation, confirmed by human evaluators.
5. Natural, low-latency spoken output.
6. Functional English → USL translation (not word-for-word substitution).
7. Avatar sign motion judged intelligible and reasonably natural by Deaf evaluators.
8. Real-time, interactive operation end-to-end.
9. Robustness under realistic Ugandan deployment conditions (variable lighting, camera quality, connectivity).
10. **Positive evaluation from the Deaf community itself** — this is the criterion that overrides all others if there is a conflict.

---

## 20. Appendix A — Glossary

- **USL:** Ugandan Sign Language.
- **Gloss:** a written label representing a sign's meaning, used for annotation (not a literal transcription of English).
- **Non-manual markers:** grammatically meaningful facial expressions, head movement, and body posture used alongside hand signs (e.g., to mark questions or negation).
- **Signer-independent split:** a train/test split ensuring no signer appears in both sets, required to measure true generalisation.
- **HamNoSys:** a phonetic notation system for describing sign-language handshapes and movements, used in some regional sign-language datasets for structured annotation.

## 21. Appendix B — Key Reference Organisations

| Organisation | Relevance |
|---|---|
| Uganda National Association of the Deaf (UNAD) | Community representation, data access, cultural review |
| Uganda National Association of Sign Language Interpreters (UNASLI) | Interpreter expertise, translation QA |
| Kyambogo University — Dept. of Hearing Impairment & Sign Language Interpretation Studies | Academic/linguistic partnership |
| National Union of Disabled Persons of Uganda (NUDIPU) | Disability-rights alignment, funding network |
| Uganda's Personal Data Protection Office (PDPO), under NITA-U | Legal compliance for data collection |
| World Federation of the Deaf (WFD) | International standards, potential funding/partnership |

## 22. Appendix C — Software Repository Structure

```
usl-translation-system/
│
├── vision/          # tracking, landmarks, preprocessing
├── recognition/      # models, training, inference
├── translation/       # sign_to_english/, english_to_sign/
├── speech/            # stt/, tts/
├── motion/            # generation, interpolation, ik
├── avatar/            # models, rendering
├── datasets/          # raw + annotated data, consent records
├── evaluation/         # metrics, human-evaluation tooling
├── api/               # FastAPI backend
├── frontend/           # React client
└── infrastructure/     # deployment, Docker, CI
```

---

*This plan should be treated as a living document, reviewed at the end of every phase with the Deaf Advisory Board and institutional partners, and revised as dataset realities, funding, and community feedback dictate.*