<div align="center">
<h1>Representation Calibration and Uncertainty Guidance for Class-Incremental Learning based on Vision Language Model</h1>
</div>

<div align="center">
<p>
    <a href="">Jiantao Tan</a><sup>1,*</sup>&nbsp;&nbsp;
    <a href="">Peixian Ma</a><sup>2,*</sup>&nbsp;&nbsp;
    <a href="">Tong Yu</a><sup>1</sup>&nbsp;&nbsp;
    <a href="">Wentao Zhang</a><sup>1</sup>&nbsp;&nbsp;
    <a href="">Ruixuan Wang</a><sup>1,3,4</sup>&nbsp;&nbsp;
</p>

<p>
    <sup>1</sup>Sun Yat-sen University
    <sup>2</sup>The Hong Kong University of Science and Technology (Guangzhou)
    <sup>3</sup>Peng Cheng Laboratory
    <sup>4</sup>Key Laboratory of Machine Intelligence and Advanced Computing
</p>
</div>


<div align="center" style="display: flex; gap: 5px; justify-content: center;">
<a href="https://arxiv.org/pdf/2512.09441"><img src="https://img.shields.io/badge/arXiv-red?style=for-the-badge&logo=arxiv"/></a>
<a href="https://github.com/MPX0222/RCUG"><img src="https://img.shields.io/badge/GitHub-black?style=for-the-badge&logo=github"/></a>
<a href="https://2026.ieeeicme.org/"><img src="https://img.shields.io/badge/ICME_2026-4b65c4?style=for-the-badge&logo=IEEE&logoColor=white"/></a>
<a href="https://github.com/MPX0222/RCUG/stargazers"><img src="https://img.shields.io/github/stars/MPX0222/VisualConcepts4CL?style=for-the-badge&color=white"/></a>
</div>

---

## 📖 Abstract

Class-incremental learning requires a learning system to continually learn knowledge of new classes and meanwhile try to preserve previously learned knowledge of old classes. As current state-of-the-art methods based on Vision-Language Models (VLMs) still suffer from the issue of differentiating classes across learning tasks. Here a novel VLM-based continual learning framework for image classification is proposed. In this framework, task-specific adapters are added to the pre-trained and frozen image encoder to learn new knowledge, and a novel cross-task representation calibration strategy based on a mixture of light-weight projectors is used to help better separate all learned classes in a unified feature space, alleviating class confusion across tasks. In addition, a novel inference strategy guided by prediction uncertainty is developed to more accurately select the most appropriate image feature for class prediction. Extensive experiments on multiple datasets under various settings demonstrate the superior performance of our method compared to existing ones. 

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{tan2025representation,
  title={Representation calibration and uncertainty guidance for class-incremental learning based on vision language model},
  author={Tan, Jiantao and Ma, Peixian and Yu, Tong and Zhang, Wentao and Wang, Ruixuan},
  booktitle={2026 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2026},
  organization={IEEE}
}
```

