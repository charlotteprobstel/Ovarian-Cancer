# Machine Learning in Ovarian Cancer Imaging: A Literature Review (2020–2026)

**Across MRI, CT, and Ultrasound Modalities**

---

## Abstract

This literature review surveys the landscape of machine learning (ML) and deep learning (DL) applied to ovarian cancer (OC) imaging across MRI, CT, and ultrasound modalities between 2020 and 2026. Drawing on 25 peer-reviewed papers spanning at least 12 countries, the review identifies key advances including the dominance of CNN-based architectures (particularly ResNet variants and U-Net families), the emergence of transformer-based models, and the growing integration of radiomics with deep learning into hybrid "radiogenomic" frameworks. Pooled diagnostic accuracies exceed 90% in many studies, yet critical barriers persist: small and institutionally siloed datasets, absence of prospective clinical validation, limited adoption of explainable AI (XAI), and virtually no regulatory-cleared products specific to ovarian cancer imaging. Despite a seven-fold increase in publication volume between 2020 and 2024, clinical translation remains embryonic. The review concludes that the field urgently requires standardised multi-centre datasets, prospective clinical trials, regulatory pathway development under the EU AI Act and FDA frameworks, and stronger collaboration between engineering and clinical teams to bridge the gap from bench to bedside.

---

## 1. Introduction

### 1.1 Goal of This Literature Review

This review aims to provide a broad, structured understanding of the current state of ML-based imaging techniques for ovarian cancer. It is written from the perspective of an early-stage PhD researcher seeking to map the field across geography, modality, and methodology. The objectives are to:

1. Catalogue the key papers, research groups, and geographic centres of activity.
2. Identify the dominant ML architectures and their reported performance.
3. Characterise persistent technical and clinical barriers.
4. Assess the degree of translation into industry and clinical practice.
5. Highlight open questions and promising directions for future research.

### 1.2 Ovarian Cancer: The Clinical Context

Ovarian cancer is a significant global health burden. According to GLOBOCAN 2022 data, approximately 324,603 women worldwide were diagnosed with ovarian cancer in 2022, with around 206,956 deaths — making it the fourth leading cause of cancer mortality among women globally (Bray et al., 2024). In the United States alone, the SEER programme estimates approximately 20,890 new cases and 12,730 deaths for 2025 (NCI SEER, 2025).

The disease is characterised by several features that make it exceptionally challenging:

- **Late-stage presentation**: Over 70% of cases are diagnosed at FIGO Stage III or IV, when the 5-year relative survival rate drops to approximately 17–31%, compared with over 90% for Stage I disease (Baldwin et al., 2012; SEER, 2025).
- **Histological heterogeneity**: Epithelial ovarian cancer (EOC) accounts for 90–95% of cases, with high-grade serous ovarian carcinoma (HGSOC) being the most common and lethal subtype (Matulonis et al., 2016).
- **No effective population-level screening**: The UK Collaborative Trial of Ovarian Cancer Screening (UKCTOCS) demonstrated that neither CA-125-based multimodal screening nor transvaginal ultrasound screening significantly reduced ovarian cancer mortality after long-term follow-up (Menon et al., 2021).
- **Rising global burden**: Annual incidence is projected to rise to nearly 500,000 by 2050, with disproportionate increases in Africa and Asia (World Ovarian Cancer Coalition, 2024).

These factors create a compelling clinical need for improved diagnostic, staging, and treatment-response prediction tools — a need that ML-based imaging approaches are increasingly positioned to address.

---

## 2. Problem Statement: ML-Driven Ovarian Cancer Diagnosis

Despite the promise of AI in medical imaging more broadly — with approximately 950 FDA-cleared AI/ML medical devices by mid-2024, the majority in radiology (FDA, 2024; Busch et al., 2024) — ovarian cancer imaging remains a conspicuously underserved domain.

The core challenges are:

1. **Imaging complexity**: Ovarian tumours are heterogeneous, often containing mixed solid and cystic components, and present in an anatomical region where soft tissue contrast on CT is limited. On MRI, the superior soft-tissue characterisation improves discrimination but at the cost of acquisition time, cost, and limited availability in low-resource settings.

2. **Subjectivity and variability**: Current diagnosis relies heavily on subjective assessment by radiologists and sonographers, with significant inter-observer variability. The O-RADS (Ovarian-Adnexal Reporting and Data System) and IOTA (International Ovarian Tumour Analysis) frameworks provide structured reporting but remain human-dependent.

3. **Multi-site, metastatic disease**: HGSOC typically presents with disseminated peritoneal disease across the pelvis, omentum, and upper abdomen, making segmentation and volumetric assessment far more complex than for localised tumours.

4. **Data scarcity and fragmentation**: Ovarian cancer is less common than breast or lung cancer, resulting in smaller, institutionally siloed datasets that are rarely shared across centres.

5. **Regulatory vacuum**: As of 2026, there is no FDA-cleared or CE-marked AI product specifically designed for ovarian cancer imaging diagnosis — a striking gap given the maturity of AI tools in breast, lung, and cardiac imaging.

This review examines how the research community has addressed these challenges, where meaningful progress has been made, and what remains to be done.

---

## 3. Papers Considered in This Review

The following table summarises the primary papers surveyed for this review, organised by year. Papers were identified through systematic searches of PubMed, IEEE Xplore, Nature, and Frontiers databases, filtered for relevance to ML/DL applied to ovarian cancer imaging via MRI, CT, or ultrasound between 2020 and 2026.

| # | Year | Title (Abbreviated) | Modality | Task | Country/Region | Journal |
|---|------|---------------------|----------|------|----------------|---------|
| 1 | 2020 | Martin-Gonzalez et al. — Integrative radiogenomics for virtual biopsy and treatment monitoring in OC | CT/MRI | Review / Framework | UK (Cambridge) | Insights Imaging |
| 2 | 2021 | Christiansen, Epstein et al. — US image analysis using DNNs for discriminating benign vs malignant ovarian tumours | US | Classification | Sweden | Ultrasound Obstet. Gynecol. |
| 3 | 2022 | Avesani et al. — CT-based radiomics and DL for BRCA mutation and PFS prediction in OC | CT | Prediction (BRCA, PFS) | Italy (Multicentric) | Cancers |
| 4 | 2022 | Wang, Perucho et al. — CT radiomics in differentiating histologic subtypes of EOC | CT | Classification | Hong Kong/China | JAMA Network Open |
| 5 | 2022 | Saida et al. — Diagnosing ovarian cancer on MRI: DL vs radiologist assessments | MRI | Classification | Japan | Cancers |
| 6 | 2022 | Xu et al. — AI performance in image-based OC identification: systematic review and meta-analysis | Multi | Meta-analysis | China | EClinicalMedicine |
| 7 | 2023 | Wang et al. — DL for ovarian lesion localisation and BOT vs EOC discrimination on MRI | MRI | Segmentation + Classification | China (Shanghai) | Scientific Reports |
| 8 | 2023 | Hu et al. — DL-based segmentation of EOC on T2W MRI | MRI | Segmentation | China (Multicentric) | Quant. Imaging Med. Surg. |
| 9 | 2023 | Wang et al. — CT-based DL segmentation of OC and radiomics feature stability | CT | Segmentation | China/Hong Kong | Quant. Imaging Med. Surg. |
| 10 | 2023 | Buddenkotte et al. — DL-based segmentation of multisite disease in OC | CT | Segmentation | UK/Germany/Italy/USA | Eur. Radiol. Exp. |
| 11 | 2023 | Crispin-Ortuzar, Woitek et al. — Integrated radiogenomics (IRON) model for NACT response | CT | Treatment response prediction | UK (Cambridge) | Nature Communications |
| 12 | 2023 | Escudero Sanchez et al. — Integrating AI tools in the clinical research setting: the OC use case | CT | Clinical pipeline integration | UK/Italy/Sweden | Diagnostics |
| 13 | 2023 | Breen et al. — AI in ovarian cancer histopathology: systematic review | Histopath. | Review | UK/Ireland | NPJ Precis. Oncol. |
| 14 | 2024 | Sadeghi et al. — DL in OC diagnosis: comprehensive review of imaging modalities | Multi | Review | Iran | Pol. J. Radiol. |
| 15 | 2024 | Du et al. — US-based DL radiomics model for differentiating benign, borderline, and malignant ovarian tumours | US | Multi-class classification | China (Guangxi) | BMC Med. Imaging |
| 16 | 2024 | El-Latif et al. — DL approach for OC detection and classification based on fuzzy DL | Histopath./MRI | Classification | Egypt | Scientific Reports |
| 17 | 2024 | Zeng et al. — Radiomics and radiogenomics: extracting more information from medical images for OC | CT/MRI | Review | China | Mil. Med. Res. |
| 18 | 2024 | Barcroft et al. — ML and radiomics for segmentation and classification of adnexal masses on US | US | Segmentation + Classification | UK (London) | NPJ Precis. Oncol. |
| 19 | 2025 | Epstein et al. — International multicenter validation of AI-driven US detection of OC | US | Classification | Sweden/International (8 countries) | Nature Medicine |
| 20 | 2025 | Gülmez — AI applications in OC detection: SLR of DL approaches and clinical translation | Multi | Systematic review | Ireland (UCD) | Crit. Rev. Oncol. Hematol. |
| 21 | 2025 | Piedimonte et al. — Predicting response to treatment and survival in advanced OC using ML and radiomics | CT/Multi | Systematic review | Canada | Cancers |
| 22 | 2025 | Chiu et al. — Advancing personalised care in OC using CT and MRI radiomics | CT/MRI | Review | UK (London) | Clinical Radiology |
| 23 | 2025 | Frontiers review — AI for OC diagnosis via US: systematic review and quantitative assessment | US | Systematic review + Meta-analysis | Italy/International | Front. Artif. Intell. |
| 24 | 2025 | Buddenkotte et al. — Multi-task DL for automatic segmentation and treatment response assessment in metastatic OC | CT | Segmentation + Response | UK (Cambridge) | Int. J. CARS |
| 25 | 2025 | Lancet Digital Health — AI in women's cancers: innovation and challenges in clinical translation | Multi | Perspective / Review | UK/International | Lancet Digital Health |

---

## 4. Advances

### 4.1 Architectural Evolution: From CNNs to Transformers

The period under review has witnessed a clear architectural progression. Early studies (2020–2022) were dominated by standard CNN architectures, particularly ResNet variants (ResNet-18, -34, -50, -101), VGG, and Inception families. These models were primarily applied to classification tasks — distinguishing benign from malignant masses, or differentiating histological subtypes.

From 2022 onward, the U-Net family (U-Net, U-Net++, nnU-Net, Attention U-Net) emerged as the dominant architecture for segmentation tasks. Wang et al. (2023) used U-Net++ with deep supervision for automated ovarian lesion segmentation on MRI, achieving a mean Dice similarity coefficient (DSC) of 0.73 on T2-weighted sagittal images. Buddenkotte et al. (2023) developed a custom architecture that outperformed the well-established nnU-Net framework for multi-site HGSOC segmentation on CT, achieving a DSC of 71 ± 20 for pelvic/ovarian and 61 ± 24 for omental lesions — performance comparable to a trainee radiologist.

The most recent wave (2024–2025) has seen the introduction of transformer-based and hybrid architectures. The landmark Nature Medicine study by Epstein et al. (2025) employed transformer-based neural network models for ovarian cancer detection on ultrasound, validated across 20 centres in eight countries using 17,119 images from 3,652 patients. This study represents the most rigorous external validation in the field to date.

### 4.2 The Radiomics–Deep Learning Convergence

A particularly notable trend has been the convergence of hand-crafted radiomics features with deep learning representations. Du et al. (2024) developed a combined deep learning radiomics (DLR) model on ultrasound for multi-class discrimination of benign, borderline, and malignant ovarian tumours, training on data from 849 patients. The combined model outperformed either approach alone.

On CT, several groups have explored this hybrid paradigm. Wang et al. (2023) demonstrated that nnU-Net-based automated segmentation of ovarian cancer on contrast-enhanced CT produced radiomics features with high stability (intraclass correlation coefficients > 0.75 for the majority of features), establishing the viability of automated radiomics pipelines.

### 4.3 Integrated Radiogenomics: The IRON Framework

Perhaps the most ambitious integration came from the Cambridge group. Crispin-Ortuzar, Woitek, Brenton et al. (2023) published in Nature Communications the IRON (Integrated Radiogenomics for Ovarian Neoadjuvant therapy) model, which combined clinical features, blood biomarkers (CA-125, ctDNA), and CT-derived radiomic features from all primary and metastatic lesion sites to predict volumetric response to neoadjuvant chemotherapy in HGSOC. Validated on an independent external cohort (n=42), the model achieved an AUC of 0.78 for RECIST 1.1 classification — substantially outperforming a clinical-only model (AUC 0.47). This work demonstrated that multi-modal, multi-scale integration is not merely additive but potentially transformative for response prediction.

### 4.4 Multi-Centre Validation at Scale

The field has begun to address the critical gap in external validation. The Epstein et al. (2025) Nature Medicine study is the gold standard in this regard: a leave-one-centre-out cross-validation across 20 centres in eight countries, demonstrating robust performance across different ultrasound systems, histological diagnoses, and patient demographics. The AI model significantly outperformed both expert and non-expert examiners and reduced simulated expert referrals by 63%.

On CT, Buddenkotte et al. (2023) used 451 scans from four institutions for their segmentation model. Avesani et al. (2022) collected data from four Italian referral centres for their BRCA/PFS prediction study — although their results were notably less optimistic, with AUCs between 0.46 and 0.59 for BRCA prediction and 0.46–0.56 for 1-year relapse, underscoring the real-world difficulty of these tasks.

### 4.5 Quantitative Summary of Model Performance

| Task | Modality | Architecture | Best Reported Performance | Key Study |
|------|----------|-------------|--------------------------|-----------|
| Benign vs malignant classification | US | Transformer-based | Outperformed expert examiners (F1, sensitivity, specificity) | Epstein et al. (2025) |
| Benign vs malignant classification | US (pooled meta-analysis, 44 studies) | Various (CNN dominant) | Mean accuracy 92.3%, AUC 0.93 | Frontiers review (2025) |
| BOT vs EOC discrimination | MRI (T2W) | U-Net++ + SE-ResNet-34 | AUC 0.87 (sagittal T2WI) | Wang et al. (2023) |
| EOC segmentation | MRI (T2W) | CNN-based | DSC up to 0.73 | Hu et al. (2023) |
| Multi-site HGSOC segmentation | CT | Custom DL model | DSC 71 ± 20 (pelvic), 61 ± 24 (omental) | Buddenkotte et al. (2023) |
| NACT response prediction | CT + blood + clinical | Ensemble ML (IRON) | AUC 0.78 (external validation) | Crispin-Ortuzar et al. (2023) |
| BRCA mutation prediction | CT (multicentric) | Radiomics + DL | AUC 0.46–0.59 (test set) | Avesani et al. (2022) |
| Multi-class tumour classification | US | DLR (CNN + Radiomics) | Best AUC per class > 0.90 | Du et al. (2024) |
| CT-based OC segmentation | CT | nnU-Net | DSC high stability for downstream radiomics | Wang et al. (2023) |

---

## 5. Drawbacks

### 5.1 Small, Siloed, and Retrospective Datasets

The most pervasive limitation across the reviewed literature is data scarcity. Sample sizes range from fewer than 100 patients (Saida et al., 2022: MRI, Japan) to several hundred (Buddenkotte et al., 2023: 451 CT scans), with only a few studies exceeding 1,000 patients. The Epstein et al. (2025) ultrasound study (3,652 patients) is a clear outlier. The Gülmez (2026) systematic review of 61 studies found that the majority used private institutional datasets, severely limiting external reproducibility.

Most studies are retrospective, single-institution, and lack temporal or geographic external validation sets. This creates a high risk of overfitting to institutional imaging protocols, scanner vendors, and patient demographics.

### 5.2 Segmentation Remains Unresolved

While classification tasks have achieved headline-grabbing accuracy figures (up to 99.7% in some studies), segmentation — the more clinically useful task for volumetric assessment, treatment planning, and response monitoring — remains far more challenging. Ovarian cancer segmentation is inherently difficult due to:

- Tumour heterogeneity (mixed solid/cystic components).
- Ill-defined boundaries, especially in the presence of ascites.
- Multi-site disease requiring segmentation of anatomically distinct lesion populations.
- Low soft-tissue contrast on CT for peritoneal deposits.

The best reported DSC values for CT-based segmentation hover around 0.61–0.71, which is adequate for research purposes but below the thresholds typically required for clinical deployment (Buddenkotte et al., 2023).

### 5.3 Explainability Deficit

Explainable AI (XAI) implementation is strikingly poor. The Gülmez (2026) systematic review found that only 7 of 61 studies implemented any form of XAI — despite growing regulatory requirements (the EU AI Act mandates transparency and human oversight for high-risk medical AI systems). Grad-CAM saliency maps are the most commonly used technique when XAI is attempted, but these provide limited clinical interpretability compared to more sophisticated methods like concept-based explanations or counterfactual reasoning.

### 5.4 Hyperparameter Optimisation Neglected

Rigorous hyperparameter tuning is conspicuously absent from most studies. The Gülmez (2026) review found that only 12 of 61 studies implemented systematic parameter optimisation. This raises concerns about the reproducibility and generalisability of reported results — it is unclear whether marginal performance gains reflect genuine architectural innovations or simply better-tuned baselines.

### 5.5 Class Imbalance and Borderline Tumours

Borderline ovarian tumours (BOTs) present a particular diagnostic challenge, as their imaging appearance overlaps substantially with both benign cysts and early malignancy. Multi-class discrimination (benign vs borderline vs malignant) remains significantly harder than binary classification, and most studies collapse the problem into a binary task. Du et al. (2024) is one of the few to tackle multi-class discrimination directly.

### 5.6 MRI Remains Understudied Relative to Ultrasound and CT

Despite MRI's superior soft-tissue characterisation, the number of ML studies using MRI for ovarian cancer lags behind those using CT and especially ultrasound. This reflects practical barriers: MRI acquisitions are slower, more expensive, and less standardised across centres. The datasets available for MRI-based ML are correspondingly smaller, and multi-centre MRI datasets for ovarian cancer are nearly non-existent in the public domain.

### 5.7 Geographic and Publication Bias

The bibliometric analysis by Xu et al. (2025) found that China led global publication output in ML for ovarian cancer with 254 articles out of 777 total. The UK (Cambridge group in particular), Italy, the USA, and Sweden are also prominent. However, there is a near-total absence of contribution from low- and middle-income countries (LMICs), where the burden of ovarian cancer is rising most steeply and where the potential impact of AI-assisted imaging would be greatest.

---

## 6. Industry Translation

### 6.1 The Gap Between Research and Regulatory Clearance

As of early 2026, no AI/ML product has received FDA 510(k) clearance or CE marking specifically for ovarian cancer imaging diagnosis. This stands in sharp contrast to breast imaging, where multiple AI products have been cleared and are commercially deployed (e.g., Kheiron's Mia, Lunit INSIGHT MMG, iCAD ProFound AI).

The FDA's database of AI/ML-enabled medical devices lists approximately 950 cleared devices by mid-2024, with radiology accounting for the vast majority. However, these are overwhelmingly concentrated in breast, lung, cardiac, and musculoskeletal imaging. Gynecological imaging, and ovarian cancer in particular, represents a conspicuous gap.

### 6.2 Adjacent Industry Activity

While no ovarian-cancer-specific products exist, several adjacent developments are relevant:

- **Cytalux (pafolacianine)**, approved by the FDA in November 2021, is a fluorescent imaging agent for intraoperative visualisation of ovarian cancer lesions. While not an AI product, it represents the closest FDA-cleared imaging innovation specific to ovarian cancer.
- **DeepHealth / Kheiron acquisition** (October 2024): RadNet's subsidiary DeepHealth acquired Kheiron Medical Technologies (London) for its Mia breast-cancer AI platform. While breast-focused, the consolidation signals the maturation of AI-assisted cancer imaging as a commercially viable space. The deal's reported low acquisition price (~$1M) also highlights the commercial challenges facing pure-play AI diagnostic companies.
- **Siemens Healthineers × DeepHealth** (December 2024): A strategic collaboration on AI-powered ultrasound operations, embedding AI informatics into imaging hardware workflows. While not ovarian-cancer-specific, this type of platform integration could eventually support ovarian mass assessment.
- **Lantern Pharma × Oregon Therapeutics** (May 2024): An AI-driven drug discovery collaboration targeting ovarian cancer treatment (XCE853 inhibitor), illustrating that AI's impact on ovarian cancer may arrive via the therapeutic pipeline before the diagnostic one.

### 6.3 Regulatory Landscape

The regulatory environment is evolving rapidly:

- **EU AI Act** (entered into force August 2024): Classifies AI medical devices as "high-risk" AI systems, imposing requirements for data governance, transparency, human oversight, bias mitigation, and post-market surveillance on top of existing MDR obligations. For radiology AI, approximately 75% of commercial devices already require notified-body conformity assessment and will therefore automatically qualify as high-risk under the AI Act (Busch et al., 2024). Healthcare AI obligations are expected to be fully enforceable by August 2027.
- **FDA PCCP Guidance** (December 2024): Finalised guidance on Predetermined Change Control Plans for AI/ML-enabled medical devices, allowing manufacturers to pre-specify how their algorithms may be updated post-approval without requiring full re-submission.
- **ESR Consensus Recommendations** (2025): The European Society of Radiology published consensus recommendations on post-market surveillance for AI medical devices in radiology, highlighting that only 29% of AI deployers consider themselves familiar with MDR/PMS regulations.

For any future ovarian-cancer-specific AI product, these overlapping regulatory frameworks (EU MDR + AI Act in Europe; FDA 510(k)/De Novo/PMA in the US) represent a significant but navigable barrier — provided that developers build clinical validation, bias assessment, and transparency into their development pipelines from the outset.

---

## 7. Conclusion

The period 2020–2026 has seen a dramatic expansion in ML-based ovarian cancer imaging research, with publication counts rising from approximately 5 papers per year in 2020–2021 to 34 in 2023–2024 (Gülmez, 2026). The field has progressed from simple CNN classifiers on small single-centre datasets to transformer-based architectures validated across 20 international centres (Epstein et al., 2025), and from isolated imaging analysis to multi-modal radiogenomic frameworks integrating imaging, genomic, and clinical data (Crispin-Ortuzar et al., 2023).

Key accomplishments include:

- **Diagnostic performance** approaching or exceeding expert radiologists for benign/malignant discrimination on ultrasound.
- **Feasibility** of automated multi-site segmentation on CT at trainee-radiologist level.
- **Proof of concept** for integrated radiogenomic treatment response prediction.

Yet the field remains fundamentally pre-clinical. No product has been translated into routine clinical use, prospective trial evidence is almost entirely absent, and the regulatory pathway remains untested for ovarian cancer imaging AI.

The most pressing needs are:

1. **Standardised, multi-centre, publicly available datasets** — ideally with harmonised acquisition protocols and expert annotations for all three modalities.
2. **Prospective clinical trials** — moving beyond retrospective validation to real-world clinical impact studies.
3. **Explainability-first design** — embedding XAI as a core requirement, not an afterthought, to meet regulatory demands and gain clinical trust.
4. **LMIC inclusion** — developing solutions deployable in resource-constrained settings where the disease burden is growing fastest.
5. **Industry engagement** — clear commercial pathways and regulatory strategies that incentivise translation from academic research to clinical products.

---

## 8. Evaluation of This Review

### 8.1 Strengths

- Covers all three major non-invasive imaging modalities (MRI, CT, ultrasound) across a six-year period.
- Includes both primary studies and recent systematic reviews/meta-analyses, providing multiple layers of evidence.
- Explicitly tracks geographic distribution of research activity.
- Addresses the industry and regulatory dimensions, which are often omitted from purely technical reviews.

### 8.2 Limitations

- **Selection bias**: Papers were identified through targeted database searches and may not capture all relevant work, particularly conference papers, non-English publications, and grey literature.
- **English-language bias**: The review is limited to English-language publications, which may underrepresent work from China (the leading country by publication volume) where some papers are published in Mandarin.
- **Publication bias**: The reviewed literature overwhelmingly reports positive results. Negative or null findings (such as the Avesani et al. (2022) study showing poor multicentric radiomics performance for BRCA prediction) are underrepresented but critically informative.
- **Exclusion of histopathology/WSI**: Per the stated scope, whole-slide image analysis was excluded, despite being a highly active parallel research track.
- **Rapid field evolution**: Given the pace of publication (accelerating through 2025–2026), some recent preprints and studies may not be captured.
- **No formal meta-analysis**: This is a narrative review; no pooled statistical analysis was performed.

### 8.3 Reflexive Note

The author's own research interest in MRI-based ovarian cancer segmentation may introduce a subtle bias toward over-weighting MRI literature and segmentation tasks. Readers should bear this in mind when interpreting the relative emphasis given to different modalities and tasks within the review.

---

## References

Avesani, G., Tran, H.E., Cammarata, G. et al. (2022). CT-based radiomics and deep learning for BRCA mutation and progression-free survival prediction in ovarian cancer using a multicentric dataset. *Cancers*, 14(11), 2739.

Barcroft, J.F., Linton-Reid, K., Landolfo, C. et al. (2024). Machine learning and radiomics for segmentation and classification of adnexal masses on ultrasound. *NPJ Precision Oncology*, 8, 41.

Bray, F., Laversanne, M., Sung, H. et al. (2024). Global cancer statistics 2022: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. *CA: A Cancer Journal for Clinicians*, 74(3), 229–263.

Breen, J. et al. (2023). Artificial intelligence in ovarian cancer histopathology: a systematic review. *NPJ Precision Oncology*, 7, 83.

Buddenkotte, T., Rundo, L., Woitek, R. et al. (2023). Deep learning-based segmentation of multisite disease in ovarian cancer. *European Radiology Experimental*, 7, 77.

Buddenkotte, T. et al. (2025). Multi-task deep learning for automatic image segmentation and treatment response assessment in metastatic ovarian cancer. *International Journal of Computer Assisted Radiology and Surgery*.

Busch, F., Kather, J.N., Johner, C. et al. (2024). Navigating the European Union Artificial Intelligence Act for Healthcare. *NPJ Digital Medicine*, 7, 210.

Chiu, S., Mascarenhas, S., Bharwani, N. et al. (2025). Advancing personalised care in ovarian cancer using CT and MRI radiomics. *Clinical Radiology*, 84, 106833.

Christiansen, F., Epstein, E. et al. (2021). Ultrasound image analysis using deep neural networks for discriminating between benign and malignant ovarian tumors. *Ultrasound in Obstetrics & Gynecology*, 57, 155–163.

Crispin-Ortuzar, M., Woitek, R., Reinius, M.A.V. et al. (2023). Integrated radiogenomics models predict response to neoadjuvant chemotherapy in high grade serous ovarian cancer. *Nature Communications*, 14, 6756.

Du, Y., Guo, W., Xiao, Y. et al. (2024). Ultrasound-based deep learning radiomics model for differentiating benign, borderline, and malignant ovarian tumours. *BMC Medical Imaging*, 24, 89.

El-Latif, E.I.A., El-dosuky, M., Darwish, A. et al. (2024). A deep learning approach for ovarian cancer detection and classification based on fuzzy deep learning. *Scientific Reports*, 14, 26463.

Epstein, E. et al. (2025). International multicenter validation of AI-driven ultrasound detection of ovarian cancer. *Nature Medicine*.

Escudero Sanchez, L., Buddenkotte, T. et al. (2023). Integrating artificial intelligence tools in the clinical research setting: the ovarian cancer use case. *Diagnostics*, 13(17), 2813.

Frontiers review (2025). Artificial intelligence for ovarian cancer diagnosis via ultrasound: a systematic review and quantitative assessment of model performance. *Frontiers in Artificial Intelligence*.

Gülmez, B. (2026). Artificial intelligence applications in ovarian cancer detection: a systematic literature review of deep learning approaches and clinical translation challenges. *Critical Reviews in Oncology/Hematology*.

Hu, D., Jian, J., Li, Y., Gao, X. (2023). Deep learning-based segmentation of epithelial ovarian cancer on T2-weighted magnetic resonance images. *Quantitative Imaging in Medicine and Surgery*, 13(3), 1464.

Martin-Gonzalez, P., Crispin-Ortuzar, M., Rundo, L. et al. (2020). Integrative radiogenomics for virtual biopsy and treatment monitoring in ovarian cancer. *Insights into Imaging*, 11, 94.

Matulonis, U.A., Sood, A.K., Fallowfield, L. et al. (2016). Ovarian cancer. *Nature Reviews Disease Primers*, 2, 16061.

Menon, U. et al. (2021). Ovarian cancer population screening and mortality after long-term follow-up in the UK Collaborative Trial of Ovarian Cancer Screening (UKCTOCS). *The Lancet*, 397(10290), 2182–2193.

Pesapane, F. et al. (2026). Artificial intelligence as medical device in radiology in 2025: the regulatory scenario in the EU, USA, and China. *European Radiology*.

Piedimonte, S. et al. (2025). Predicting response to treatment and survival in advanced ovarian cancer using machine learning and radiomics: a systematic review. *Cancers*, 17, 336.

Sadeghi, M.H. et al. (2024). Deep learning in ovarian cancer diagnosis: a comprehensive review of various imaging modalities. *Polish Journal of Radiology*, 89, e30–e48.

Saida, T. et al. (2022). Diagnosing ovarian cancer on MRI: a preliminary study comparing deep learning and radiologist assessments. *Cancers*, 14(4), 987.

Wang, Y., Zhang, H. et al. (2023). Deep learning for the ovarian lesion localization and discrimination between borderline and malignant ovarian tumors based on routine MR imaging. *Scientific Reports*, 13, 2770.

Wang, Y., Wang, M. et al. (2023). CT-based deep learning segmentation of ovarian cancer and the stability of the extracted radiomics features. *Quantitative Imaging in Medicine and Surgery*, 13(8).

Xu, H.L. et al. (2022). Artificial intelligence performance in image-based ovarian cancer identification: systematic review and meta-analysis. *EClinicalMedicine*, 53, 101662.

Zeng, S., Wang, X.L., Yang, H. (2024). Radiomics and radiogenomics: extracting more information from medical images for the diagnosis and prognostic prediction of ovarian cancer. *Military Medical Research*, 11, 77.
