# Pore Identification

## 1. Introduction & Purpose

This document consolidates our research into facial pores — their biology, how they're visually identified (both by humans and algorithmically), the existing academic and commercial work in this space, and our current thinking on how we might approach building a pore-identification system of our own.

It covers what pores are, what they're commonly confused with, what's already been done in computer vision for this problem, what data is available to us, and where our own approach currently stands.

## 2. Pores

### 2.1 Introduction

Pores are tiny openings in the skin that serve as the entry and exit points for sweat and oil. Each pore is surrounded by a layer of skin cells and is connected to the underlying dermis. The size and appearance of pores can vary significantly from person to person.

### 2.2 Architecture

There are two types of pores: hair follicles and sweat pores.

- **Hair follicles (sebaceous pores)** — connected to sebaceous glands and produce sebum, an oily substance that helps lubricate the skin. These are the pores we usually mean when talking about visible "pores" on the face.
- **Sweat pores** — connected to sweat glands and involved in sweating. Not really visible to the naked eye, and largely irrelevant to what people mean by "enlarged" or "clogged" pores.

Sebaceous pore size is directly tied to the size of the underlying sebaceous gland — larger glands produce more sebum and sit under a wider follicular opening, which is part of why pore size varies so much between individuals and between facial zones. Density is also uneven across the face: sebaceous glands (and therefore visible pores) are concentrated far more heavily in the T-zone and butterfly zone (nose, cheeks, forehead) than on the jawline or temples, which is why CV research in this space consistently crops to those regions rather than the full face.

### 2.3 Functionality

Pores' main tasks are producing sebum and sweat. They also allow the skin to "breathe" — i.e., they support gas and moisture exchange at the skin surface alongside their secretory role.

### 2.4 What Affects Pore Appearance

Pore visibility isn't fixed — it fluctuates based on a mix of genetic and environmental factors:

- **Genetics** — baseline sebaceous gland size and skin type are largely inherited.
- **Sebum production** — oilier skin stretches the follicular opening more visibly; hormonal shifts (puberty, menstrual cycle) directly affect sebum output.
- **Aging and collagen loss** — as skin loses elasticity, pores lose the structural support that keeps them tight, making them appear larger and less able to "snap back."
- **Sun damage** — UV exposure degrades collagen and elastin around the follicle, compounding the aging effect above.
- **Dead skin buildup and dehydration** — both can make pore openings appear more pronounced or shadowed, even without any actual size change.

### 2.5 Related Diseases

- **Acne** — hair follicles clogged with sebum, dead skin cells, and bacteria.
- **Enlarged pores** — associated with oily and combination skin types; not a disease.
- **Clogged pores** — excess oil, dirt, and dead skin clog up a pore, forming blackheads or whiteheads; can lead to acne.

### 2.6 Treatment

- **Topical treatments** — retinoids, benzoyl peroxide, etc.
- **Professional treatment** — laser, chemical peels, etc.

### 2.7 Preventive Measures

- **Skincare routine** — cleansing, moisturizing, toning.
- **Diet and lifestyle** — a balanced diet rich in antioxidants, vitamins, and minerals can promote healthy skin. Staying hydrated is also crucial for maintaining skin elasticity and preventing clogged pores.

## 3. Pore Identification

This section covers how pores are identified visually — by a human eye — and, critically, how they're distinguished from conditions that are commonly mistaken for them. This distinction matters: if we want to identify pores reliably, we also need to understand what isn't a pore, since several benign conditions get misdiagnosed as pore-related issues (and vice versa) even by casual observation.

### 3.1 How Pores Are Visually Identified

A normal pore, on close inspection, is small, shallow, and appears in dense numbers rather than as an isolated feature — especially across the nose and cheeks. There's no raised structure, no distinct halo, and no plug of material that differs meaningfully from the surrounding skin surface. This is the baseline against which the following look-alikes are distinguished.

### 3.2 Common Look-Alikes and How They Differ

- **Dilated Pore of Winer (DPW)** — often mistaken for a large blackhead or an unusually enlarged single pore. The giveaway is that it appears as a solitary lesion (not part of a cluster like normal pores), is noticeably larger, and — under dermoscopy — shows a distinct central dark core surrounded by a pale halo and a pigmented rim, a layered structure normal pores never show.
- **Milia** — mistaken for clogged pores or whiteheads, typically spotted around the eyes and cheeks. The difference: milia are small, hard, closed cysts under the skin with no actual opening — unlike a clogged pore, they can't be extracted the way a blackhead or whitehead can.
- **Trichostasis Spinulosa** — mistaken for a single large clogged pore. In reality, it's multiple fine hairs bundled and trapped within one follicle, appearing as a tiny dark tufted spot, mainly on the nose and cheeks.
- **Sebaceous Filaments** — not a disease at all, but constantly confused with clogged pores or blackheads. They're a normal, structural part of every pore (a grayish-yellowish lining of sebum and skin cells), refill quickly after extraction, and are present in essentially everyone — unlike a clogged pore, which is an actual buildup issue.

### 3.3 Additional Hard-Negative Classes

Our own research (DPW, milia, trichostasis spinulosa, sebaceous filaments) covers the conditions most frequently confused with pores in general dermatology writing. A broader internal reference ontology also flags additional structures that any production system should reject. Folding those in:

- **Open/closed comedones** — dark or raised follicular plugs — visually close to a clogged pore but structurally distinct (a plug vs. an open pore).
- **Hair-bearing follicles / stubble** — a dark point with a visible shaft or directional shadow, easily mistaken for a pore at low resolution.
- **Ice-pick / atrophic scars** — small, deep-looking depressions that read as an oversized pore but are actually scar tissue.
- **Fine wrinkles / furrows** — thin linear shadows that can be misclassified as elongated pores if the model isn't shape-aware.
- **Dry flakes / product residue** — bright irregular particles with edge shadow — a common false positive on unwashed or made-up skin.
- **Pigmented spots / freckles** — dark, compact features with no actual opening geometry.
- **Image artifacts** — compression ringing, oversharpening halos, motion blur, and beauty-filter smoothing can all fabricate or erase apparent pores — these need to be rejected at the image level, not the pore level.

> This section is our current stand-in for a formal annotation ontology — it is accurate as a reference list, but has not been reviewed by a dermatologist or tested for inter-rater agreement.

### 3.4 Why This Matters

Any identification method — human or automated — that claims to "identify pores" is incomplete if it can't also rule out these look-alikes. Flagging DPW as an enlarged pore, or treating sebaceous filaments as a clogged-pore problem, isn't actually solving the identification problem — it's pattern-matching on surface similarity without the differential knowledge that separates a true pore from its counterparts.

## 4. Skin Beauty CV

Computer Vision (CV) is the field of AI concerned with getting computers to interpret visual information — detecting, segmenting, and classifying patterns in images. In skincare specifically, CV is used to analyze facial images and flag concerns like wrinkles, pigmentation, texture, and pore visibility, largely as a way to power personalized product recommendations.

A few examples of products already working in this space:

- **L'Oréal Skin Genius** — probably the most polished commercial example currently available, trained on L'Oréal's own proprietary dataset. Positioned as a consumer-facing skin analysis tool feeding directly into product recommendations.
- **L'Oréal SkinConsult AI / Perso (via ModiFace)** — uses deep learning trained on large sets of facial images to evaluate wrinkles, texture, pigmentation, and enlarged pores from user-uploaded selfies, also factoring in environmental data.

These are useful reference points for what a "finished" product in this space looks like, even though none of them publish their underlying methodology or datasets.

## 5. Pore Identification via CV

We currently have four sources on this topic, two of which are paywalled and inaccessible to us in full (see References). The two accessible papers are broken down below.

### 5.1 PS-Net Paper

PS-Net formulates pore segmentation as a two-stage, weakly supervised task — with no human annotation required at any point.

- **Warm-up stage** — uses Difference of Gaussian (DoG), a classical filter, to find an initial batch of pores. DoG is conservative — it isn't great at catching smaller or harder-to-see pores, but when it does flag something, it's almost always a genuine pore. This high-precision/low-recall behavior is exactly what makes it usable as a starting label source despite being incomplete.
- **Maturity stage** — using attention mechanisms and additional techniques (exact method and code not publicly disclosed), the model is trained on this initial data until it effectively "fixes itself" — refining and expanding its own predictions beyond what DoG alone was able to catch.
- **Additional architectural detail** — the high-frequency attention module and Bayesian shape module — still needs to be filled in more thoroughly; this section requires a bit more research before it's complete.

### 5.2 MDPI Paper

The first study to tackle simultaneous identification of both pores and wrinkles.

**Core concepts**
- **Zero Padding** — a padding trick that leaks positional information into CNN feature maps, letting the network learn "where" a feature sits.
- **Location-Aware Attention** — an attention map biased toward known typical zones instead of scanning the whole face equally.
- **Model**: U-Net (encoder-decoder) alongside two attention-type schemes.

**Ground truth generation — 3 steps**
1. Based on U-Net, attention mechanisms are added to refine feature maps.
2. A feature map is added to increase the detail captured for wrinkles/pores.
3. Localization is improved via zero padding + location-aware attention.

**Semi-automatic labeling process**
1. Rough manual labeling of the wrinkle.
2. The input is modified with a wrinkle-enhancing filter.
3. The two images are multiplied together.
4. (Special formulas for texture map implementation are on page 4 of the paper.)

Why location matters here: human faces have very similar structures, allowing them to be parsed into consistent parts. Unlike acne or rosacea — which can appear anywhere — pore and wrinkle locations are largely predetermined (pores mostly on the nose and cheeks, the "butterfly zone"). Because of this, the CNN's positional understanding is further reinforced, since the areas it needs to check are inherently limited.

**Dataset**: 314 images, quite small and easily reproducible. Cropped to focus on the eye, forehead, and butterfly zones, then resized to 768×640. Roughly 84% train / 16% validation split.

**Training**: Input is H×W×C (C = RGB channels), output is a grayscale H×W×2 map.

**Core contributions**: Zero padding inputs, location-aware learning, skin-specific ground truth generation.

**Limitations**: No severity analysis yet, imperfect ground truth, basic loss function.

## 6. Our Approach

The following represents our preliminary methodology rather than a finalized strategy. We are currently weighing two primary execution paths:

### Option A — Implement PS-Net or MDPI directly

- **PS-Net** — A strong candidate if our priority is eliminating manual annotation. It uses classical DoG filtering for initial coarse labeling followed by self-training to bootstrap from unlabeled sources like Kaggle. However, the lack of public code means custom components like the Bayesian shape module and high-frequency attention must be built from the ground up, introducing technical risk.
- **MDPI** — Offers a more straightforward replication path using a standard U-Net with zero-padding and spatial attention. The main drawbacks are the labor-intensive semi-automatic labeling required and the limited size of their original dataset (314 images), which may struggle to generalize without pretraining.

This would take a lot of time, and effort to research further. So we went with:

### Option B — Train a segmentation model on Roboflow

* Process unlabeled data through the pretrained Segmentation Pore model to generate automated masks, then use those masks to bootstrap our own trained model (Model 1 → Model 2, Section 8) rather than relying on the pretrained model alone.
* Even though a meaningful portion of the training data carried noisy or incomplete labels (a mix of merged multi-source datasets, see the "0"/"pore" schema issue in Section 8.1), the resulting model reached decent real-world segmentation quality — a result only visible after manually reviewing raw predictions rather than trusting the evaluation metrics alone (Section 8.3).
* **Confidence threshold behavior — the key finding:** at higher confidence (~50%), the model behaves like PS-Net's DoG warm-up stage: it fires rarely, but almost always correctly when it does. What's notable is that this held up much further than expected — lowering the confidence threshold all the way down to ~10% still produced reliably correct detections on manual review, not the flood of false positives a "trade precision for recall" threshold drop would normally imply. In other words, the model doesn't need to stay conservative to stay trustworthy.
* **Practical implication:** this means the model can be run at a low confidence threshold for broad candidate coverage during bootstrap segmentation, with only light spot-check monitoring rather than a full manual review pass per image — a meaningfully cheaper labeling workflow than initially assumed, since the earlier plan (Section 6.5) treated high confidence as the only "safe" operating point.
* **Caveat:** this finding comes from manual spot-checking across test images, not a formal precision/recall re-evaluation against a cleaned ground truth. It should be treated the same way as the Section 8.3 finding — a strong qualitative signal that changes our workflow, not yet a validated number to put in a metrics table.

**Pre Process Pipeline**:
1. Isolate the butterfly zone (cheeks/forehead), excluding eyes, mouth, and bridge of the nose.
2. Standardize image resolution across the dataset.
3. Dedupe the data set
4. Segment and Upload the dataset for annotations

**Supporting Tools & Libraries**: The repo that was built for preprocessing tasks mostly to learn how to do it, will be updated as required.

[Preprocessing](https://github.com/Kostimizasyon/bluesense_pores)

[Inference](https://github.com/bluesense-ai/smart-beauty-pores)

7. Dataset
Neither PS-Net nor the MDPI paper make their datasets publicly available, so we have to rely on public data instead. A few sources we've found so far:

* Roboflow — 1,104 labeled images of individual pores — [pores1-5-2](https://universe.roboflow.com/projects-3qaaz/pores1-5-2)
* Roboflow — 558 labeled images of individual pores — [seg-pore](https://universe.roboflow.com/foodrecog-meto9/seg-pore/browse)
* Roboflow — 297 labeled images of individual pores — [skin-pore](https://universe.roboflow.com/foodrecog-meto9/skin-pore)
* Roboflow — labeled images of individual pores (v6) — [pores_datasets](https://universe.roboflow.com/dermascan-ai/pores_datasets-e3gmv/dataset/6)
* Roboflow — labeled images of individual pores (v3) — [pores-a7nzb](https://universe.roboflow.com/cool-it/pores-a7nzb/dataset/3)
* Roboflow — labeled images of individual pores — [pores1_2](https://universe.roboflow.com/projects-3qaaz/pores1_2)
* Kaggle — 600 unlabeled pore images + acne/pigmentation/wrinkle data — [facial-skin-acne-pigmentation-pores-wrinkles](https://www.kaggle.com/datasets/shijo96john/facial-skin-acne-pigmentation-pores-wrinkles)

## 8. Conclusion

Reliable pore identification, whether by eye or algorithmically, depends on also recognizing the conditions commonly mistaken for pores (DPW, milia, trichostasis spinulosa, sebaceous filaments). PS-Net and the MDPI paper offer two solid but different approaches to the CV side — automated coarse labeling plus self-training versus positional priors on a small, reproducible dataset. Commercial tools like L'Oréal's confirm the space is viable, though methods aren't shared. Our own pipeline is only roughly sketched, model choice is still open, and we still need to close the dataset gap around look-alike conditions before results can be meaningfully evaluated.

## 9. References

**Pore biology & dermatology**
- BHSkin Dermatology — Pore glossary/explainer — https://ai.bhskin.com/glossary/pore-dermatology-explained

**Commercial products**
- L'Oréal Skin Genius — https://www.lorealparisusa.com/skin-genius-landing-page

**Academic papers — pore-specific (paywalled)**
- IEEE Xplore — https://ieeexplore.ieee.org/document/8517224
- IEEE Xplore — https://ieeexplore.ieee.org/document/9867133

**Academic papers — accessible**
- PS-Net (Visual Intelligence, Springer) — https://link.springer.com/article/10.1007/s44267-025-00088-9
- MDPI Diagnostics — pore & wrinkle segmentation — https://www.mdpi.com/2075-4418/13/11/1894

**Look-alike conditions**
- Dermoscopic Characterization of Dilated Pore of Winer — https://journals.lww.com/cddr/fulltext/2019/03010/dermoscopic_characterization_of_dilated_pore_of.18.aspx
- StatPearls / NCBI Bookshelf — Dilated Pore of Winer — https://www.ncbi.nlm.nih.gov/books/NBK532967/
- Medscape / eMedicine — Dilated Pore of Winer — https://emedicine.medscape.com/article/1061188-overview
