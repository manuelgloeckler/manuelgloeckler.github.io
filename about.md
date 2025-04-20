---
layout: article
titles:
  # @start locale config
  en      : &EN       About
  en-GB   : *EN
  en-US   : *EN
  en-CA   : *EN
  en-AU   : *EN
  zh-Hans : &ZH_HANS  关于
  zh      : *ZH_HANS
  zh-CN   : *ZH_HANS
  zh-SG   : *ZH_HANS
  zh-Hant : &ZH_HANT  關於
  zh-TW   : *ZH_HANT
  zh-HK   : *ZH_HANT
  ko      : &KO       소개
  ko-KR   : *KO
  fr      : &FR       À propos
  fr-BE   : *FR
  fr-CA   : *FR
  fr-CH   : *FR
  fr-FR   : *FR
  fr-LU   : *FR
  # @end locale config
key: page-about
---

<table class="table-no-border">
  <tr>
    <td style="width: 500px;">
     <div>
  {{
  "
I am a PhD student at the University of Tübingen and part of the International Max-Planck Research School for Intelligent Systems (IMPRS-IS). My supervisor is Prof. Dr. Jakob H. Macke.

I am developing machine learning tools to perform Bayesian inference for simulation-based models. I am interested in the intersection of machine learning, statistics and science applications. Some of my main interests are:

- Bayesian inference
- Neural density estimation and generative models
- Probabilistic and differentiable programming
- Uncertainty quantification/calibration

" | markdownify}}
</div>
    </td>
    <td style="width: 400px;">
      <div class="image-text-container">
  <img src="assets/image.jpg" alt="Your Name" width="400">
</div>
    </td>
  </tr>
</table>

## Curriculum Vitae

For a nicer and likely more complete overview, here is a [CV](/assets/academic_cv.pdf) as PDF.

### Education

* **PHD student at the International Max-Planck Research School for Intelligent Systems** [(IMPRS-IS)](https://imprs.is.mpg.de/)\
  Apr 2022 - Now\
  Supervised by Prof. Dr. Jakob Macke, University Tübingen.


* **Master of Science in Bioinformatics**\
  Okt 2019 - Mar 2022\
  University Tübingen (Grade: 1.15, 3.9 GPA equivalent)\
  Thesis: ["Variational methods for simulation-based inference"](assets/FINAL_thesis_version.pdf)\
  Transcript: [Transcript of Records](assets/master_tor.pdf), [Certificate](assets/master_zeugniss.pdf)\
  Transcript (GPA):[Grade summary](assets/list_of_grades.pdf)

* **Bachelor of Science in Bioinformatics**\
  Okt 2016 - Sep 2019\
  University Tübingen (Grade: 1.31, 3.7 GPA equivalent)\
  Thesis:  ["The landscapes of CD8+ T cell immunogenicity from a self-tolerance based perspective in sequence space"](assets/bachelor_thesis.pdf)\
  Transcript: [Transcript of Records](assets/backelor_tor.pdf), [Certificate](assets/bachelor_zeugniss.pdf)\
  Transcript (GPA):[Grade summary](assets/list_of_grades.pdf)

* **A-Levels**\
  Joachim-Hahn-Gymnasium, Blaubeuren, Germany (Grade: 2.1, 3.0 GPA equivalent)

### Work experience

* **Research assistant**\
  Seq 2020 - Feb 2022\
  University Tübingen, Computational Systems Biology, Junior Prof. Dr. Andreas Dräger\
  Supervised by Dr. Reihaneh Mostolizadeh.

* **Student assistant**\
  Okt 2018 - Feb 2019\
  University Tübingen, Theory of Machine Learning Group, Prof. Dr. Ulrike von Luxburg.\
  Teaching assistant for lecture "Algorithms".

### Selected Publications

For a full list of publications please refer to [google scholar](https://scholar.google.com/citations?user=0Vdv0H0AAAAJ&hl=de).

* **All-in-one simulation-based inference**, ICML 2024\
  Manuel Glöckler, Michael Deistler, Christian Weilbach, Frank Wood, Jakob H Macke [[arxiv]](https://arxiv.org/abs/2404.09636)

* **Variational methods for simulation-based inference**, ICLR 2022\
  Manuel Glöckler, Michael Deistler, Jakob H Macke [[arxiv]](https://arxiv.org/abs/2203.04176)

* **Adversarial robustness of amortized Bayesian inference**, ICML 2023\
  Manuel Glöckler, Michael Deistler, Jakob H Macke [[arxiv]](https://arxiv.org/abs/2305.14984)

### Other

*  IOP trusted reviewer [[certificate]](assets/iop_trusted_reviewer.pdf)
*  Cambridge ELLIS Machine Learning Summer School and poster presentation 2022 [[certificate]](assets/ellis_summer_school.pdf)
*  Machine Learning Summer School 2021 [[certificate]](assets/mlls_summer_school.pdf)


## Some notes

If I read into interesting topics I typically as default write a small article about it together with a way simplified example (if it is a method). This is mostly what I will post here. I try to explain it from scratch and try to include (simple) proof (ideas) for any claims I raise, but I assume from the reader some basic knowledge about math and statistics.
