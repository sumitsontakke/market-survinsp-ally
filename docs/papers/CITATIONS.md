# Citations

<!-- audit-workstream-A: preprints withdrawn 2026-08-06 -->

> **Note (2026-08-06).** Both preprint PDFs have been temporarily
> withdrawn pending post-audit revision (see [`../AUDIT.md`](../AUDIT.md) and [`../REMEDIATION_PLAN.md`](../REMEDIATION_PLAN.md)). The BibTeX entries below still reference
> the withdrawn PDFs; **do not cite them in new work** until the
> revised versions are published. The software DOI at
> `10.5281/zenodo.21755230` is unaffected and remains citable.


If you use this project, please cite **the software** (Zenodo DOI) and,
if your work engages with either paper's specific framing, also cite
the paper.

## Software

```bibtex
@software{sontakke2026msa,
  author    = {Sontakke, Sumit and Joshi, Milan},
  title     = {{market-survinsp-ally}: Graph-neural-network market surveillance toolkit for collusive-trading detection},
  year      = {2026},
  month     = aug,
  publisher = {Zenodo},
  version   = {v0.1.0},
  doi       = {10.5281/zenodo.21755230},
  url       = {https://doi.org/10.5281/zenodo.21755230}
}
```

DOI landing page: [https://doi.org/10.5281/zenodo.21755230](https://doi.org/10.5281/zenodo.21755230)

Each future release gets its own version DOI. The Zenodo "all versions"
DOI on the record page always resolves to the latest.

## Paper 1 — ML / graph-learning framing

**Feature-Augmented Graph Neural Networks for Out-of-Distribution
Detection of Collusive Market Manipulation: A Multi-Cohort Study.**

Preprint PDF: [`model_journey_paper.pdf`](model_journey_paper.pdf) ·
LaTeX source: [`model_journey_paper.tex`](model_journey_paper.tex)

```bibtex
@article{sontakke2026msa_paper1,
  author  = {Sontakke, Sumit and Joshi, Milan},
  title   = {{Feature-Augmented Graph Neural Networks for Out-of-Distribution Detection of Collusive Market Manipulation: A Multi-Cohort Study}},
  year    = {2026},
  journal = {Under submission (arXiv preprint forthcoming)},
  url     = {https://github.com/sumitsontakke/market-survinsp-ally/blob/main/docs/papers/model_journey_paper.pdf}
}
```

**Cite this paper if you're working on:** OOD generalisation for graph
neural networks, feature-augmented GNNs, family-shift robustness in
learned detectors, or tier-2 stacked classifier architectures.

## Paper 2 — RegTech / applied-AI framing

**Trade Market Manipulation Detection with Graph Neural Networks:
A Calibrated-Synthesis-to-Deployment Surveillance Pipeline.**

Preprint PDF: [`model_journey_paper_v2.pdf`](model_journey_paper_v2.pdf) ·
LaTeX source: [`model_journey_paper_v2.tex`](model_journey_paper_v2.tex)

```bibtex
@article{sontakke2026msa_paper2,
  author  = {Sontakke, Sumit and Joshi, Milan},
  title   = {{Trade Market Manipulation Detection with Graph Neural Networks: A Calibrated-Synthesis-to-Deployment Surveillance Pipeline}},
  year    = {2026},
  journal = {Under submission (arXiv preprint forthcoming)},
  url     = {https://github.com/sumitsontakke/market-survinsp-ally/blob/main/docs/papers/model_journey_paper_v2.pdf}
}
```

**Cite this paper if you're working on:** market-surveillance systems,
regulator-facing tooling, calibrated-synthesis pipelines for training
detectors under scarce real-world label data, or the end-to-end
"synthesis → detection → investigation" arc.

Once the arXiv preprints post (pending endorsement from Dr Milan Joshi),
this file will be updated with arXiv IDs, and BibTeX entries will
include `eprint`, `archivePrefix`, and `primaryClass` fields.

## Substrate

If your work relies specifically on the ABIDES cross-generator results
(paper table row 2, and Section 5 of Paper 2), also cite the ABIDES
simulator:

```bibtex
@article{byrd2020abides,
  author  = {Byrd, David and Hybinette, Maria and Balch, Tucker Hybinette},
  title   = {{ABIDES}: Towards {H}igh-{F}idelity {M}ulti-{A}gent {M}arket {S}imulation},
  year    = {2020},
  journal = {Proceedings of the 2020 ACM SIGSIM Conference on Principles of Advanced Discrete Simulation (PADS)}
}
```

## Related but not required

If you cite the manipulation-signature feature set (φ₁ through φ₆) as
a stand-alone contribution, cite Paper 1 — that's where the feature
family is derived and their discriminative power on the OOD cohort is
tabulated.
