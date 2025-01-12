
from dotenv import load_dotenv
load_dotenv()

import sys
import os
import pandas as pd
from typing import List, Dict, Any, Optional, Set
import csv
import json
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import textwrap

import networkx as nx
import matplotlib.pyplot as plt
import uuid

from unstructured.partition.pdf import partition_pdf
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model = "gpt-4o",
    temperature = 0,
    max_retries = 3
)

sys_prommpt = """
# System Prompt: ML Conference Recommendation System

You are a knowledgeable assistant specialized in helping researchers determine the most suitable conference for their machine learning papers. Use the following guidelines to analyze papers and recommend appropriate venues.

## Conference Profiles

### NEURIPS (Neural Information Processing Systems)
- **Core Focus**: Broad machine learning, AI, theoretical contributions
- **Ideal Papers**:
  - Strong theoretical foundations with mathematical rigor
  - Novel algorithms with comprehensive theoretical analysis
  - Interdisciplinary work connecting ML with neuroscience, statistics, or physics
  - Cutting-edge research in deep learning, reinforcement learning, and foundation models
  - Papers addressing social impact, fairness, and ethical considerations in ML
- **Key Requirements**:
  - Exceptional technical depth and mathematical rigor
  - Novel theoretical contributions or significant algorithmic innovations
  - Comprehensive empirical validation
  - Clear articulation of broader impact
- **Not Suitable For**:
  - Purely application-focused papers without theoretical insights
  - Incremental improvements without significant novelty
  - Papers lacking rigorous theoretical analysis

### KDD (Knowledge Discovery and Data Mining)
- **Core Focus**: Data science applications, scalability, real-world impact
- **Ideal Papers**:
  - Novel data mining algorithms with clear practical applications
  - Scalable solutions for big data challenges
  - Industrial applications with significant impact
  - Innovations in recommender systems, graph mining, or time series analysis
  - Papers demonstrating real-world deployments
- **Key Requirements**:
  - Clear demonstration of practical impact
  - Strong empirical evaluation on real-world datasets
  - Scalability analysis
  - Business or industrial relevance
- **Not Suitable For**:
  - Purely theoretical papers without practical applications
  - Small-scale experiments without scalability considerations
  - Papers focusing solely on mathematical proofs

### CVPR (Computer Vision and Pattern Recognition)
- **Core Focus**: Computer vision, visual understanding, pattern recognition
- **Ideal Papers**:
  - Novel computer vision algorithms and architectures
  - Advances in 3D vision, image processing, or video analysis
  - Vision-language models and multimodal learning
  - Applications in autonomous driving, robotics, or medical imaging
- **Key Requirements**:
  - Strong technical innovation in computer vision
  - Comprehensive visual results and demonstrations
  - Thorough experimental evaluation
  - Clear advancement over state-of-the-art
- **Not Suitable For**:
  - Papers without significant visual components
  - General ML papers not focused on vision
  - Pure theoretical work without visual applications

### EMNLP (Empirical Methods in Natural Language Processing)
- **Core Focus**: Natural language processing, empirical methods
- **Ideal Papers**:
  - Novel NLP models and architectures
  - Empirical studies of language phenomena
  - Advances in machine translation, dialogue systems, or text analysis
  - Language model analysis and evaluation
- **Key Requirements**:
  - Strong empirical evaluation methodology
  - Clear contributions to NLP research
  - Thorough analysis of results
  - Consideration of linguistic phenomena
- **Not Suitable For**:
  - Papers without significant NLP components
  - Pure theoretical work without empirical validation
  - Applications where language is not central

### TMLR (Transactions on Machine Learning Research)
- **Core Focus**: Technical correctness, methodological rigor
- **Ideal Papers**:
  - Technically sound ML research across all areas
  - Thorough empirical studies
  - Novel theoretical frameworks
  - Reproducible research with comprehensive analysis
- **Key Requirements**:
  - Impeccable technical correctness
  - Comprehensive experimental validation
  - Clear methodology
  - Strong reproducibility focus
- **Not Suitable For**:
  - Rushed or preliminary results
  - Papers without thorough validation
  - Work lacking clear methodology

## Analysis Framework

When evaluating a paper, consider these aspects:

1. **Research Domain**
   - Primary field (ML, Data Mining, Computer Vision, NLP)
   - Theoretical vs. Applied focus
   - Interdisciplinary connections

2. **Technical Depth**
   - Theoretical foundations
   - Mathematical rigor
   - Empirical methodology
   - Experimental validation

3. **Impact and Applications**
   - Practical relevance
   - Scalability considerations
   - Real-world applications
   - Industry potential

4. **Innovation Level**
   - Novelty of contribution
   - Advancement over state-of-the-art
   - Potential impact on field

5. **Presentation and Validation**
   - Experimental thoroughness
   - Result analysis depth
   - Reproducibility considerations
   - Visual components

## Decision Process

1. First, identify the paper's primary domain and research focus
2. Assess the balance between theoretical and practical contributions
3. Evaluate technical depth and validation methodology
4. Consider the paper's potential impact and audience
5. Match these characteristics with conference profiles

## Red Flags

Watch for these warning signs that might indicate a mismatch:
- Domain mismatch (e.g., pure NLP paper to CVPR)
- Depth mismatch (e.g., applied paper without theory to NEURIPS)
- Validation mismatch (e.g., small-scale evaluation to KDD)
- Impact mismatch (e.g., theoretical paper without applications to KDD)

## Example Recommendations

1. **Theoretical ML Paper**
   - Primary: NEURIPS
   - Secondary: TMLR
   - Avoid: KDD

2. **Industrial Application Paper**
   - Primary: KDD
   - Secondary: Domain-specific (CVPR/EMNLP)
   - Avoid: NEURIPS (unless strong theoretical component)

3. **Computer Vision Innovation**
   - Primary: CVPR
   - Secondary: NEURIPS (if theoretical)
   - Avoid: EMNLP/KDD

4. **NLP Research**
   - Primary: EMNLP
   - Secondary: NEURIPS (if theoretical)
   - Avoid: CVPR

5. **Methodological Study**
   - Primary: TMLR
   - Secondary: Domain-specific conference
   - Consider: Based on specific focus

## Response Format

When making recommendations, provide:
1. Primary recommendation with detailed justification
2. Alternative venues if applicable
3. Specific strengths and weaknesses for each suggested venue
4. Potential improvements to better fit target venues
"""

content_r007 = """
Advancements in 3D Food Modeling: A Review of the MetaFood Challenge Techniques and Outcomes

Abstract

The growing focus on leveraging computer vision for dietary oversight and nutri- tion tracking has spurred the creation of sophisticated 3D reconstruction methods for food. The lack of comprehensive, high-fidelity data, coupled with limited collaborative efforts between academic and industrial sectors, has significantly hindered advancements in this domain. This study addresses these obstacles by introducing the MetaFood Challenge, aimed at generating precise, volumetrically accurate 3D food models from 2D images, utilizing a checkerboard for size cal- ibration. The challenge was structured around 20 food items across three levels of complexity: easy (200 images), medium (30 images), and hard (1 image). A total of 16 teams participated in the final assessment phase. The methodologies developed during this challenge have yielded highly encouraging outcomes in 3D food reconstruction, showing great promise for refining portion estimation in dietary evaluations and nutritional tracking. Further information on this workshop challenge and the dataset is accessible via the provided URL.1 Introduction

The convergence of computer vision technologies with culinary practices has pioneered innovative approaches to dietary monitoring and nutritional assessment. The MetaFood Workshop Challenge represents a landmark initiative in this emerging field, responding to the pressing demand for precise and scalable techniques for estimating food portions and monitoring nutritional consumption. Such technologies are vital for fostering healthier eating behaviors and addressing health issues linked to diet.

By concentrating on the development of accurate 3D models of food derived from various visual inputs, including multiple views and single perspectives, this challenge endeavors to bridge the disparity between current methodologies and practical needs. It promotes the creation of unique solutions capable of managing the intricacies of food morphology, texture, and illumination, while also meeting the real-world demands of dietary evaluation. This initiative gathers experts from computer vision, machine learning, and nutrition science to propel 3D food reconstruction technologies forward. These advancements have the potential to substantially enhance the precision and utility of food portion estimation across diverse applications, from individual health tracking to extensive nutritional investigations.

Conventional methods for assessing diet, like 24-Hour Recall or Food Frequency Questionnaires (FFQs), are frequently reliant on manual data entry, which is prone to inaccuracies and can be burdensome. The lack of 3D data in 2D RGB food images further complicates the use of regression- based methods for estimating food portions directly from images of eating occasions. By enhancing 3D reconstruction for food, the aim is to provide more accurate and intuitive nutritional assessment tools. This technology could revolutionize the sharing of culinary experiences and significantly impact nutrition science and public health.Participants were tasked with creating 3D models of 20 distinct food items from 2D images, mim- icking scenarios where mobile devices equipped with depth-sensing cameras are used for dietary

recording and nutritional tracking. The challenge was segmented into three tiers of difficulty based on the number of images provided: approximately 200 images for easy, 30 for medium, and a single top-view image for hard. This design aimed to rigorously test the adaptability and resilience of proposed solutions under various realistic conditions. A notable feature of this challenge was the use of a visible checkerboard for physical referencing and the provision of depth images for each frame, ensuring the 3D models maintained accurate real-world measurements for portion size estimation.

This initiative not only expands the frontiers of 3D reconstruction technology but also sets the stage for more reliable and user-friendly real-world applications, including image-based dietary assessment. The resulting solutions hold the potential to profoundly influence nutritional intake monitoring and comprehension, supporting broader health and wellness objectives. As progress continues, innovative applications are anticipated to transform personal health management, nutritional research, and the wider food industry. The remainder of this report is structured as follows: Section 2 delves into the existing literature on food portion size estimation, Section 3 describes the dataset and evaluation framework used in the challenge, and Sections 4, 5, and 6 discuss the methodologies and findings of the top three teams (VoIETA, ININ-VIAUN, and FoodRiddle), respectively.2 Related Work

Estimating food portions is a crucial part of image-based dietary assessment, aiming to determine the volume, energy content, or macronutrients directly from images of meals. Unlike the well-studied task of food recognition, estimating food portions is particularly challenging due to the lack of 3D information and physical size references necessary for accurately judging the actual size of food portions. Accurate portion size estimation requires understanding the volume and density of food, elements that are hard to deduce from a 2D image, underscoring the need for sophisticated techniques to tackle this problem. Current methods for estimating food portions are grouped into four categories.

Stereo-Based Approaches use multiple images to reconstruct the 3D structure of food. Some methods estimate food volume using multi-view stereo reconstruction based on epipolar geometry, while others perform two-view dense reconstruction. Simultaneous Localization and Mapping (SLAM) has also been used for continuous, real-time food volume estimation. However, these methods are limited by their need for multiple images, which is not always practical.

Model-Based Approaches use predefined shapes and templates to estimate volume. For instance, certain templates are assigned to foods from a library and transformed based on physical references to estimate the size and location of the food. Template matching approaches estimate food volume from a single image, but they struggle with variations in food shapes that differ from predefined templates. Recent work has used 3D food meshes as templates to align camera and object poses for portion size estimation.Depth Camera-Based Approaches use depth cameras to create depth maps, capturing the distance from the camera to the food. These depth maps form a voxel representation used for volume estimation. The main drawback is the need for high-quality depth maps and the extra processing required for consumer-grade depth sensors.

Deep Learning Approaches utilize neural networks trained on large image datasets for portion estimation. Regression networks estimate the energy value of food from single images or from an "Energy Distribution Map" that maps input images to energy distributions. Some networks use both images and depth maps to estimate energy, mass, and macronutrient content. However, deep learning methods require extensive data for training and are not always interpretable, with performance degrading when test images significantly differ from training data.

While these methods have advanced food portion estimation, they face limitations that hinder their widespread use and accuracy. Stereo-based methods are impractical for single images, model-based approaches struggle with diverse food shapes, depth camera methods need specialized hardware, and deep learning approaches lack interpretability and struggle with out-of-distribution samples. 3D reconstruction offers a promising solution by providing comprehensive spatial information, adapting to various shapes, potentially working with single images, offering visually interpretable results, and enabling a standardized approach to food portion estimation. These benefits motivated the organization of the 3D Food Reconstruction challenge, aiming to overcome existing limitations and

develop more accurate, user-friendly, and widely applicable food portion estimation techniques, impacting nutritional assessment and dietary monitoring.

3 Datasets and Evaluation Pipeline3.1 Dataset Description

The dataset for the MetaFood Challenge features 20 carefully chosen food items from the MetaFood3D dataset, each scanned in 3D and accompanied by video recordings. To ensure precise size accuracy in the reconstructed 3D models, each food item was captured alongside a checkerboard and pattern mat, serving as physical scaling references. The challenge is divided into three levels of difficulty, determined by the quantity of 2D images provided for reconstruction:

¢ Easy: Around 200 images taken from video.

* Medium: 30 images.

¢ Hard: A single image from a top-down perspective.

Table 1 details the food items included in the dataset.

Table 1: MetaFood Challenge Data Details

Object Index Food Item Difficulty Level Number of Frames 1 Strawberry Easy 199 2 Cinnamon bun Easy 200 3 Pork rib Easy 200 4 Corn Easy 200 5 French toast Easy 200 6 Sandwich Easy 200 7 Burger Easy 200 8 Cake Easy 200 9 Blueberry muffin Medium 30 10 Banana Medium 30 11 Salmon Medium 30 12 Steak Medium 30 13 Burrito Medium 30 14 Hotdog Medium 30 15 Chicken nugget Medium 30 16 Everything bagel Hard 1 17 Croissant Hard 1 18 Shrimp Hard 1 19 Waffle Hard 1 20 Pizza Hard 1

3.2 Evaluation Pipeline

The evaluation process is split into two phases, focusing on the accuracy of the reconstructed 3D models in terms of shape (3D structure) and portion size (volume).

3.2.1 Phase-I: Volume Accuracy

In the first phase, the Mean Absolute Percentage Error (MAPE) is used to evaluate portion size accuracy, calculated as follows:

12 MAPE = — > i=l Ai — Fi Aj x 100% qd)

where A; is the actual volume (in ml) of the i-th food item obtained from the scanned 3D food mesh, and F; is the volume calculated from the reconstructed 3D mesh.3.2.2 Phase-II: Shape Accuracy

Teams that perform well in Phase-I are asked to submit complete 3D mesh files for each food item. This phase involves several steps to ensure precision and fairness:

* Model Verification: Submitted models are checked against the final Phase-I submissions for consistency, and visual inspections are conducted to prevent rule violations.

* Model Alignment: Participants receive ground truth 3D models and a script to compute the final Chamfer distance. They must align their models with the ground truth and prepare a transformation matrix for each submitted object. The final Chamfer distance is calculated using these models and matrices.

¢ Chamfer Distance Calculation: Shape accuracy is assessed using the Chamfer distance metric. Given two point sets X and Y,, the Chamfer distance is defined as:

4 > 1 2 dev(X.Y) = 15 Do mip lle — yll2 + Ty DL main lle — all (2) EX yey

This metric offers a comprehensive measure of similarity between the reconstructed 3D models and the ground truth. The final ranking is determined by combining scores from both Phase-I (volume accuracy) and Phase-II (shape accuracy). Note that after the Phase-I evaluation, quality issues were found with the data for object 12 (steak) and object 15 (chicken nugget), so these items were excluded from the final overall evaluation.

4 First Place Team - VoIETA

4.1 Methodology

The team’s research employs multi-view reconstruction to generate detailed food meshes and calculate precise food volumes.4.1.1 Overview

The team’s method integrates computer vision and deep learning to accurately estimate food volume from RGBD images and masks. Keyframe selection ensures data quality, supported by perceptual hashing and blur detection. Camera pose estimation and object segmentation pave the way for neural surface reconstruction, creating detailed meshes for volume estimation. Refinement steps, including isolated piece removal and scaling factor adjustments, enhance accuracy. This approach provides a thorough solution for accurate food volume assessment, with potential uses in nutrition analysis.4.1.2 The Team’s Proposal: VoIETA

The team starts by acquiring input data, specifically RGBD images and corresponding food object masks. The RGBD images, denoted as Ip = {Ip;}"_,, where n is the total number of frames, provide depth information alongside RGB images. The food object masks, {uf }"_,, help identify regions of interest within these images.

Next, the team selects keyframes. From the set {Ip;}7_1, keyframes {If }4_, C {Ipi}f_4 are chosen. A method is implemented to detect and remove duplicate and blurry images, ensuring high-quality frames. This involves applying a Gaussian blurring kernel followed by the fast Fourier transform method. Near-Image Similarity uses perceptual hashing and Hamming distance threshold- ing to detect similar images and retain overlapping ones. Duplicates and blurry images are excluded to maintain data integrity and accuracy.

Using the selected keyframes {I if }*_ |, the team estimates camera poses through a method called PixSfM, which involves extracting features using SuperPoint, matching them with SuperGlue, and refining them. The outputs are the camera poses {Cj} Ro crucial for understanding the scene’s spatial layout.

In parallel, the team uses a tool called SAM for reference object segmentation. SAM segments the reference object with a user-provided prompt, producing a reference object mask /" for each keyframe. This mask helps track the reference object across all frames. The XMem++ method extends the reference object mask /¥ to all frames, creating a comprehensive set of reference object masks {/?}"_,. This ensures consistent reference object identification throughout the dataset.

To create RGBA images, the team combines RGB images, reference object masks {M/??}"_,, and food object masks {/}"}"_,. This step, denoted as {J}? }"~,, integrates various data sources into a unified format for further processing.The team converts the RGBA images {I/*}?_, and camera poses {C}}4_, into meaningful metadata and modeled data D,,,. This transformation facilitates accurate scene reconstruction.

The modeled data D,,, is input into NeuS2 for mesh reconstruction. NeuS2 generates colorful meshes {R/, R"} for the reference and food objects, providing detailed 3D representations. The team uses the "Remove Isolated Pieces" technique to refine the meshes. Given that the scenes contain only one food item, the diameter threshold is set to 5% of the mesh size. This method deletes isolated connected components with diameters less than or equal to 5%, resulting in a cleaned mesh {RC , RC’}. This step ensures that only significant parts of the mesh are retained.

The team manually identifies an initial scaling factor S using the reference mesh via MeshLab. This factor is fine-tuned to Sy using depth information and food and reference masks, ensuring accurate scaling relative to real-world dimensions. Finally, the fine-tuned scaling factor S, is applied to the cleaned food mesh RCS, producing the final scaled food mesh RF. This step culminates in an accurately scaled 3D representation of the food object, enabling precise volume estimation.4.1.3 Detecting the scaling factor

Generally, 3D reconstruction methods produce unitless meshes by default. To address this, the team manually determines the scaling factor by measuring the distance for each block of the reference object mesh. The average of all block lengths [ay is calculated, while the actual real-world length is constant at ca; = 0.012 meters. The scaling factor S = lpeat /lavg is applied to the clean food mesh RC! , resulting in the final scaled food mesh RFS in meters.

The team uses depth information along with food and reference object masks to validate the scaling factors. The method for assessing food size involves using overhead RGB images for each scene. Initially, the pixel-per-unit (PPU) ratio (in meters) is determined using the reference object. Subse- quently, the food width (f,,,) and length (f7) are extracted using a food object mask. To determine the food height (f;,), a two-step process is followed. First, binary image segmentation is performed using the overhead depth and reference images, yielding a segmented depth image for the reference object. The average depth is then calculated using the segmented reference object depth (d,-). Similarly, employing binary image segmentation with an overhead food object mask and depth image, the average depth for the segmented food depth image (d+) is computed. The estimated food height f), is the absolute difference between d, and dy. To assess the accuracy of the scaling factor S, the food bounding box volume (f,, x fi x fn) x PPU is computed. The team evaluates if the scaling factor S' generates a food volume close to this potential volume, resulting in S'sjn¢. Table 2 lists the scaling factors, PPU, 2D reference object dimensions, 3D food object dimensions, and potential volume.For one-shot 3D reconstruction, the team uses One-2-3-45 to reconstruct a 3D model from a single RGBA view input after applying binary image segmentation to both food RGB and mask images. Isolated pieces are removed from the generated mesh, and the scaling factor S', which is closer to the potential volume of the clean mesh, is reused.

4.2 Experimental Results

4.2.1 Implementation settings

Experiments were conducted using two GPUs: GeForce GTX 1080 Ti/12G and RTX 3060/6G. The Hamming distance for near image similarity was set to 12. For Gaussian kernel radius, even numbers in the range [0...30] were used for detecting blurry images. The diameter for removing isolated pieces was set to 5%. NeuS2 was run for 15,000 iterations with a mesh resolution of 512x512, a unit cube "aabb scale" of 1, "scale" of 0.15, and "offset" of [0.5, 0.5, 0.5] for each food scene.4.2.2 VolETA Results

The team extensively validated their approach on the challenge dataset and compared their results with ground truth meshes using MAPE and Chamfer distance metrics. The team’s approach was applied separately to each food scene. A one-shot food volume estimation approach was used if the number of keyframes k equaled 1; otherwise, a few-shot food volume estimation was applied. Notably, the keyframe selection process chose 34.8% of the total frames for the rest of the pipeline, showing the minimum frames with the highest information.

Table 2: List of Extracted Information Using RGBD and Masks

Level Id Label Sy PPU Ry x Ri (fw x fi x fh) 1 Strawberry 0.08955223881 0.01786 320 x 360 = (238 x 257 x 2.353) 2 Cinnamon bun 0.1043478261 0.02347 236 x 274 = (363 x 419 x 2.353) 3 Pork rib 0.1043478261 0.02381 246x270 (435 x 778 x 1.176) Easy 4 Corn 0.08823529412 0.01897 291 x 339 (262 x 976 x 2.353) 5 French toast 0.1034482759 0.02202 266 x 292 (530 x 581 x 2.53) 6 Sandwich 0.1276595745 0.02426 230 x 265 (294 x 431 x 2.353) 7 Burger 0.1043478261 0.02435 208 x 264 (378 x 400 x 2.353) 8 Cake 0.1276595745 0.02143. 256 x 300 = (298 x 310 x 4.706) 9 Blueberry muffin —_0.08759124088 0.01801 291x357 (441 x 443 x 2.353) 10 Banana 0.08759124088 0.01705 315x377 (446 x 857 x 1.176) Medium 11 Salmon 0.1043478261 0.02390 242 x 269 (201 x 303 x 1.176) 13 Burrito 0.1034482759 0.02372 244 x 27 (251 x 917 x 2.353) 14 Frankfurt sandwich —_0.1034482759 0.02115. 266 x 304. (400 x 1022 x 2.353) 16 Everything bagel —_0.08759124088 0.01747 306 x 368 = (458 x 134 x 1.176 ) ) Hard 17 Croissant 0.1276595745 0.01751 319 x 367 = (395 x 695 x 2.176 18 Shrimp 0.08759124088 0.02021 249x318 (186 x 95 x 0.987) 19 Waffle 0.01034482759 0.01902 294 x 338 (465 x 537 x 0.8) 20 Pizza 0.01034482759 0.01913 292 x 336 (442 x 651 x 1.176)After finding keyframes, PixSfM estimated the poses and point cloud. After generating scaled meshes, the team calculated volumes and Chamfer distance with and without transformation metrics. Meshes were registered with ground truth meshes using ICP to obtain transformation metrics.

Table 3 presents quantitative comparisons of the team’s volumes and Chamfer distance with and without estimated transformation metrics from ICP. For overall method performance, Table 4 shows the MAPE and Chamfer distance with and without transformation metrics.

Additionally, qualitative results on one- and few-shot 3D reconstruction from the challenge dataset are shown. The model excels in texture details, artifact correction, missing data handling, and color adjustment across different scene parts.

Limitations: Despite promising results, several limitations need to be addressed in future work:

¢ Manual processes: The current pipeline includes manual steps like providing segmentation prompts and identifying scaling factors, which should be automated to enhance efficiency.

¢ Input requirements: The method requires extensive input information, including food masks and depth data. Streamlining these inputs would simplify the process and increase applicability.

* Complex backgrounds and objects: The method has not been tested in environments with complex backgrounds or highly intricate food objects.

¢ Capturing complexities: The method has not been evaluated under different capturing complexities, such as varying distances and camera speeds.

¢ Pipeline complexity: For one-shot neural rendering, the team currently uses One-2-3-45. They aim to use only the 2D diffusion model, Zero123, to reduce complexity and improve efficiency.

Table 3: Quantitative Comparison with Ground Truth Using Chamfer DistanceL Id Team’s Vol. GT Vol. Ch. w/tm Ch. w/otm 1 40.06 38.53 1.63 85.40 2 216.9 280.36 TA2 111.47 3 278.86 249.67 13.69 172.88 E 4 279.02 295.13 2.03 61.30 5 395.76 392.58 13.67 102.14 6 205.17 218.44 6.68 150.78 7 372.93 368.77 4.70 66.91 8 186.62 73.13 2.98 152.34 9 224.08 232.74 3.91 160.07 10 153.76 63.09 2.67 138.45 M ill 80.4 85.18 3.37 151.14 13 363.99 308.28 5.18 147.53 14 535.44 589.83 4.31 89.66 16 163.13 262.15 18.06 28.33 H 17 224.08 81.36 9.44 28.94 18 25.4 20.58 4.28 12.84 19 110.05 08.35 11.34 23.98 20 130.96 19.83 15.59 31.05

Table 4: Quantitative Comparison with Ground Truth Using MAPE and Chamfer Distance

MAPE Ch. w/t.m Ch. w/o t.m (%) sum mean sum mean 10.973 0.130 0.007 1.715 0.095

5 Second Place Team - ININ-VIAUN

5.1 Methodology

This section details the team’s proposed network, illustrating the step-by-step process from original images to final mesh models.5.1.1 Scale factor estimation

The procedure for estimating the scale factor at the coordinate level is illustrated in Figure 9. The team adheres to a method involving corner projection matching. Specifically, utilizing the COLMAP dense model, the team acquires the pose of each image along with dense point cloud data. For any given image img, and its extrinsic parameters [R|t];,, the team initially performs threshold-based corner detection, setting the threshold at 240. This step allows them to obtain the pixel coordinates of all detected corners. Subsequently, using the intrinsic parameters k and the extrinsic parameters [R|t],, the point cloud is projected onto the image plane. Based on the pixel coordinates of the corners, the team can identify the closest point coordinates P* for each corner, where i represents the index of the corner. Thus, they can calculate the distance between any two corners as follows:

Di =(PE- PPP Wid G 6)

To determine the final computed length of each checkerboard square in image k, the team takes the minimum value of each row of the matrix D” (excluding the diagonal) to form the vector d*. The median of this vector is then used. The final scale calculation formula is given by Equation 4, where 0.012 represents the known length of each square (1.2 cm):

0.012 scale = —,——__~ 4 eae Ss med) ”5.1.2 3D Reconstruction

The 3D reconstruction process, depicted in Figure 10, involves two different pipelines to accommodate variations in input viewpoints. The first fifteen objects are processed using one pipeline, while the last five single-view objects are processed using another.

For the initial fifteen objects, the team uses COLMAP to estimate poses and segment the food using the provided segment masks. Advanced multi-view 3D reconstruction methods are then applied to reconstruct the segmented food. The team employs three different reconstruction methods: COLMAP, DiffusioNeRF, and NeRF2Mesh. They select the best reconstruction results from these methods and extract the mesh. The extracted mesh is scaled using the estimated scale factor, and optimization techniques are applied to obtain a refined mesh.

For the last five single-view objects, the team experiments with several single-view reconstruction methods, including Zero123, Zerol23++, One2345, ZeroNVS, and DreamGaussian. They choose ZeroNVS to obtain a 3D food model consistent with the distribution of the input image. The intrinsic camera parameters from the fifteenth object are used, and an optimization method based on reprojection error refines the extrinsic parameters of the single camera. Due to limitations in single-view reconstruction, depth information from the dataset and the checkerboard in the monocular image are used to determine the size of the extracted mesh. Finally, optimization techniques are applied to obtain a refined mesh.5.1.3 Mesh refinement

During the 3D Reconstruction phase, it was observed that the model’s results often suffered from low quality due to holes on the object’s surface and substantial noise, as shown in Figure 11.

To address the holes, MeshFix, an optimization method based on computational geometry, is em- ployed. For surface noise, Laplacian Smoothing is used for mesh smoothing operations. The Laplacian Smoothing method adjusts the position of each vertex to the average of its neighboring vertices:

1 (new) __ y7(old) (old) (old) anes (Cee JEN (i)

In their implementation, the smoothing factor X is set to 0.2, and 10 iterations are performed.

5.2. Experimental Results

5.2.1 Estimated scale factor

The scale factors estimated using the described method are shown in Table 5. Each image and the corresponding reconstructed 3D model yield a scale factor, and the table presents the average scale factor for each object.

5.2.2 Reconstructed meshes

The refined meshes obtained using the described methods are shown in Figure 12. The predicted model volumes, ground truth model volumes, and the percentage errors between them are presented in Table 6.5.2.3. Alignment

The team designs a multi-stage alignment method for evaluating reconstruction quality. Figure 13 illustrates the alignment process for Object 14. First, the central points of both the predicted and ground truth models are calculated, and the predicted model is moved to align with the central point of the ground truth model. Next, ICP registration is performed for further alignment, significantly reducing the Chamfer distance. Finally, gradient descent is used for additional fine-tuning to obtain the final transformation matrix.

The total Chamfer distance between all 18 predicted models and the ground truths is 0.069441 169.

Table 5: Estimated Scale Factors

Object Index Food Item Scale Factor 1 Strawberry 0.060058 2 Cinnamon bun 0.081829 3 Pork rib 0.073861 4 Corn 0.083594 5 French toast 0.078632 6 Sandwich 0.088368 7 Burger 0.103124 8 Cake 0.068496 9 Blueberry muffin 0.059292 10 Banana 0.058236 11 Salmon 0.083821 13 Burrito 0.069663 14 Hotdog 0.073766

Table 6: Metric of Volume

Object Index Predicted Volume Ground Truth Error Percentage 1 44.51 38.53 15.52 2 321.26 280.36 14.59 3 336.11 249.67 34.62 4 347.54 295.13 17.76 5 389.28 392.58 0.84 6 197.82 218.44 9.44 7 412.52 368.77 11.86 8 181.21 173.13 4.67 9 233.79 232.74 0.45 10 160.06 163.09 1.86 11 86.0 85.18 0.96 13 334.7 308.28 8.57 14 517.75 589.83 12.22 16 176.24 262.15 32.77 17 180.68 181.36 0.37 18 13.58 20.58 34.01 19 117.72 108.35 8.64 20 117.43 119.83 20.03

6 Best 3D Mesh Reconstruction Team - FoodRiddle6.1 Methodology

To achieve high-fidelity food mesh reconstruction, the team developed two procedural pipelines as depicted in Figure 14. For simple and medium complexity cases, they employed a structure-from- motion strategy to ascertain the pose of each image, followed by mesh reconstruction. Subsequently, a sequence of post-processing steps was implemented to recalibrate the scale and improve mesh quality. For cases involving only a single image, the team utilized image generation techniques to facilitate model generation.

6.1.1 Multi- View Reconstruction

For Structure from Motion (SfM), the team enhanced the advanced COLMAP method by integrating SuperPoint and SuperGlue techniques. This integration significantly addressed the issue of limited keypoints in scenes with minimal texture, as illustrated in Figure 15.

In the mesh reconstruction phase, the team’s approach builds upon 2D Gaussian Splatting, which employs a differentiable 2D Gaussian renderer and includes regularization terms for depth distortion

and normal consistency. The Truncated Signed Distance Function (TSDF) results are utilized to produce a dense point cloud.

During post-processing, the team applied filtering and outlier removal methods, identified the outline of the supporting surface, and projected the lower mesh vertices onto this surface. They utilized the reconstructed checkerboard to correct the model’s scale and employed Poisson reconstruction to create a complete, watertight mesh of the subject.6.1.2 Single-View Reconstruction

For 3D reconstruction from a single image, the team utilized advanced methods such as LGM, Instant Mesh, and One-2-3-45 to generate an initial mesh. This initial mesh was then refined in conjunction with depth structure information.

To adjust the scale, the team estimated the object’s length using the checkerboard as a reference, assuming that the object and the checkerboard are on the same plane. They then projected the 3D object back onto the original 2D image to obtain a more precise scale for the object.

6.2. Experimental Results

Through a process of nonlinear optimization, the team sought to identify a transformation that minimizes the Chamfer distance between their mesh and the ground truth mesh. This optimization aimed to align the two meshes as closely as possible in three-dimensional space. Upon completion of this process, the average Chamfer dis- tance across the final reconstructions of the 20 objects amounted to 0.0032175 meters. As shown in Table 7, Team FoodRiddle achieved the best scores for both multi- view and single-view reconstructions, outperforming other teams in the competition.

Table 7: Total Errors for Different Teams on Multi-view and Single-view Data

Team Multi-view (1-14) Single-view (16-20) FoodRiddle 0.036362 0.019232 ININ-VIAUN 0.041552 0.027889 VolETA 0.071921 0.0587267 Conclusion

This report examines and compiles the techniques and findings from the MetaFood Workshop challenge on 3D Food Reconstruction. The challenge sought to enhance 3D reconstruction methods by concentrating on food items, tackling the distinct difficulties presented by varied textures, reflective surfaces, and intricate geometries common in culinary subjects.

The competition involved 20 diverse food items, captured under various conditions and with differing numbers of input images, specifically designed to challenge participants in creating robust reconstruc- tion models. The evaluation was based on a two-phase process, assessing both portion size accuracy through Mean Absolute Percentage Error (MAPE) and shape accuracy using the Chamfer distance metric.

Of all participating teams, three reached the final submission stage, presenting a range of innovative solutions. Team VolETA secured first place with the best overall performance in both Phase-I and Phase-II, followed by team ININ-VIAUN in second place. Additionally, the FoodRiddle team exhibited superior performance in Phase-II, highlighting a competitive and high-caliber field of entries for 3D mesh reconstruction. The challenge has successfully advanced the field of 3D food reconstruction, demonstrating the potential for accurate volume estimation and shape reconstruction in nutritional analysis and food presentation applications. The novel methods developed by the participating teams establish a strong foundation for future research in this area, potentially leading to more precise and user-friendly approaches for dietary assessment and monitoring.

10
"""
content_r009 = """
The Importance of Written Explanations in
Aggregating Crowdsourced Predictions
Abstract
This study demonstrates that incorporating the written explanations provided by
individuals when making predictions enhances the accuracy of aggregated crowd-
sourced forecasts. The research shows that while majority and weighted vote
methods are effective, the inclusion of written justifications improves forecast
accuracy throughout most of a question’s duration, with the exception of its final
phase. Furthermore, the study analyzes the attributes that differentiate reliable and
unreliable justifications.
1 Introduction
The concept of the "wisdom of the crowd" posits that combining information from numerous non-
expert individuals can produce answers that are as accurate as, or even more accurate than, those
provided by a single expert. A classic example of this concept is the observation that the median
estimate of an ox’s weight from a large group of fair attendees was remarkably close to the actual
weight. While generally supported, the idea is not without its limitations. Historical examples
demonstrate instances where crowds behaved irrationally, and even a world chess champion was able
to defeat the combined moves of a crowd.
In the current era, the advantages of collective intelligence are widely utilized. For example, Wikipedia
relies on the contributions of volunteers, and community-driven question-answering platforms have
garnered significant attention from the research community. When compiling information from
large groups, it is important to determine whether the individual inputs were made independently. If
not, factors like group psychology and the influence of persuasive arguments can skew individual
judgments, thus negating the positive effects of crowd wisdom.
This paper focuses on forecasts concerning questions spanning political, economic, and social
domains. Each forecast includes a prediction, estimating the probability of a particular event, and
a written justification that explains the reasoning behind the prediction. Forecasts with identical
predictions can have justifications of varying strength, which, in turn, affects the perceived reliability
of the predictions. For instance, a justification that simply refers to an external source without
explanation may appear to rely heavily on the prevailing opinion of the crowd and might be considered
weaker than a justification that presents specific, verifiable facts from external resources.
To clarify the terminology used: a "question" is defined as a statement that seeks information (e.g.,
"Will new legislation be implemented before a certain date?"). Questions have a defined start and
end date, and the period between these dates constitutes the "life" of the question. "Forecasters"
are individuals who provide a "forecast," which consists of a "prediction" and a "justification." The
prediction is a numerical representation of the likelihood of an event occurring. The justification
is the text provided by the forecaster to support their prediction. The central problem addressed in
this work is termed "calling a question," which refers to the process of determining a final prediction
by aggregating individual forecasts. Two strategies are employed for calling questions each day
throughout their life: considering forecasts submitted on the given day ("daily") and considering the
last forecast submitted by each forecaster ("active").
Inspired by prior research on recognizing and fostering skilled forecasters, and analyzing written
justifications to assess the quality of individual or collective forecasts, this paper investigates the
automated calling of questions throughout their duration based on the forecasts available each day.
The primary contributions are empirical findings that address the following research questions:
* When making a prediction on a specific day, is it advantageous to include forecasts from previous
days? (Yes) * Does the accuracy of the prediction improve when considering the question itself
and the written justifications provided with the forecasts? (Yes) * Is it easier to make an accurate
prediction toward the end of a question’s duration? (Yes) * Are written justifications more valuable
when the crowd’s predictions are less accurate? (Yes)
In addition, this research presents an examination of the justifications associated with both accurate
and inaccurate forecasts. This analysis aims to identify the features that contribute to a justification
being more or less credible.
2 Related Work
The language employed by individuals is indicative of various characteristics. Prior research includes
both predictive models (using language samples to predict attributes about the author) and models
that provide valuable insights (using language samples and author attributes to identify differentiating
linguistic features). Previous studies have examined factors such as gender and age, political ideology,
health outcomes, and personality traits. In this paper, models are constructed to predict outcomes
based on crowd-sourced forecasts without knowledge of individual forecasters’ identities.
Previous research has also explored how language use varies depending on the relationships between
individuals. For instance, studies have analyzed language patterns in social networks, online commu-
nities, and corporate emails to understand how individuals in positions of authority communicate.
Similarly, researchers have examined how language provides insights into interpersonal interactions
and relationships. In terms of language form and function, prior research has investigated politeness,
empathy, advice, condolences, usefulness, and deception. Related to the current study’s focus,
researchers have examined the influence of Wikipedia editors and studied influence levels within
online communities. Persuasion has also been analyzed from a computational perspective, including
within the context of dialogue systems. The work presented here complements these previous studies.
The goal is to identify credible justifications to improve the aggregation of crowdsourced forecasts,
without explicitly targeting any of the aforementioned characteristics.
Within the field of computational linguistics, the task most closely related to this research is argumen-
tation. A strong justification for a forecast can be considered a well-reasoned supporting argument.
Previous work in this area includes identifying argument components such as claims, premises,
backing, rebuttals, and refutations, as well as mining arguments that support or oppose a particular
claim. Despite these efforts, it was found that crowdsourced justifications rarely adhere to these
established argumentation frameworks, even though such justifications are valuable for aggregating
forecasts.
Finally, several studies have focused on forecasting using datasets similar or identical to the one used
in this research. From a psychological perspective, researchers have explored strategies for enhancing
forecasting accuracy, such as utilizing top-performing forecasters (often called "superforecasters"),
and have analyzed the traits that contribute to their success. These studies aim to identify and cultivate
superforecasters but do not incorporate the written justifications accompanying forecasts. In contrast,
the present research develops models to call questions without using any information about the
forecasters themselves. Within the field of computational linguistics, researchers have evaluated the
language used in high-quality justifications, focusing on aspects like rating, benefit, and influence.
Other researchers have developed models to predict forecaster skill using the textual justifications
from specific datasets, such as the Good Judgment Open data, and have also applied these models
to predict the accuracy of individual forecasts in other contexts, such as company earnings reports.
However, none of these prior works have specifically aimed to call questions throughout their entire
duration.
2
3 Dataset
The research utilizes data from the Good Judgment Open, a platform where questions are posted, and
individuals submit their forecasts. The questions primarily revolve around geopolitics, encompassing
areas such as domestic and international politics, the economy, and social matters. For this study, all
binary questions were collected, along with their associated forecasts, each comprising a prediction
and a justification. In total, the dataset contains 441 questions and 96,664 forecasts submitted
over 32,708 days. This dataset significantly expands upon previous research, nearly doubling the
number of forecasts analyzed. Since the objective is to accurately call questions throughout their
entire duration, all forecasts with written justifications are included, regardless of factors such as
justification length or the number of forecasts submitted by a single forecaster. Additionally, this
approach prioritizes privacy, as no information about the individual forecasters is utilized.
Table 1: Analysis of the questions from our dataset. Most questions are relatively long, contain two
or more named entities, and are open for over one month.
Metric Min Q1 Q2 (Median) Q3 Max Mean
# tokens 8 16 20 28 48 21.94
# entities 0 2 3 5 11 3.47
# verbs 0 2 2 3 6 2.26
# days open 2 24 59 98 475 74.16
Table 1 provides a basic analysis of the questions in the dataset. The majority of questions are
relatively lengthy, containing more than 16 tokens and multiple named entities, with geopolitical,
person, and date entities being the most frequent. In terms of duration, half of the questions remain
open for nearly two months, and 75% are open for more than three weeks.
An examination of the topics covered by the questions using Latent Dirichlet Allocation (LDA)
reveals three primary themes: elections (including terms like "voting," "winners," and "candidate"),
government actions (including terms like "negotiations," "announcements," "meetings," and "passing
(a law)"), and wars and violent crimes (including terms like "groups," "killing," "civilian (casualties),"
and "arms"). Although not explicitly represented in the LDA topics, the questions address both
domestic and international events within these broad themes.
Table 2: Analysis of the 96,664 written justifications submitted by forecasters in our dataset. The
readability scores indicate that most justifications are easily understood by high school students (11th
or 12th grade), although a substantial amount (>25%) require a college education (Flesch under 50 or
Dale-Chall over 9.0).
Min Q1 Q2 Q3 Max
#sentences 1 1 1 3 56
#tokens 1 10 23 47 1295
#entities 0 0 2 4 154
#verbs 0 1 3 6 174
#adverbs 0 0 1 3 63
#adjectives 0 0 2 4 91
#negation 0 0 1 3 69
Sentiment -2.54 0 0 0.20 6.50
Readability
Flesch -49.68 50.33 65.76 80.62 121.22
Dale-Chall 0.05 6.72 7.95 9.20 19.77
Table 2 presents a fundamental analysis of the 96,664 forecast justifications in the dataset. The median
length is relatively short, consisting of one sentence and 23 tokens. Justifications mention named
entities less frequently than the questions themselves. Interestingly, half of the justifications contain
at least one negation, and 25% include three or more. This suggests that forecasters sometimes base
their predictions on events that might not occur or have not yet occurred. The sentiment polarity of
3
the justifications is generally neutral. In terms of readability, both the Flesch and Dale-Chall scores
suggest that approximately a quarter of the justifications require a college-level education for full
comprehension.
Regarding verbs and nouns, an analysis using WordNet lexical files reveals that the most common
verb classes are "change" (e.g., "happen," "remain," "increase"), "social" (e.g., "vote," "support,"
"help"), "cognition" (e.g., "think," "believe," "know"), and "motion" (e.g., "go," "come," "leave").
The most frequent noun classes are "act" (e.g., "election," "support," "deal"), "communication" (e.g.,
"questions," "forecast," "news"), "cognition" (e.g., "point," "issue," "possibility"), and "group" (e.g.,
"government," "people," "party").
4 Experiments and Results
Experiments are conducted to address the challenge of accurately calling a question throughout
its duration. The input consists of the question itself and the associated forecasts (predictions and
justifications), while the output is an aggregated answer to the question derived from all forecasts.
The number of instances corresponds to the total number of days all questions were open. Both
simple baselines and a neural network are employed, considering both (a) daily forecasts and (b)
active forecasts submitted up to ten days prior.
The questions are divided into training, validation, and test subsets. Subsequently, all forecasts
submitted throughout the duration of each question are assigned to their respective subsets. It’s
important to note that randomly splitting the forecasts would be an inappropriate approach. This is
because forecasts for the same question submitted on different days would be distributed across the
training, validation, and test subsets, leading to data leakage and inaccurate performance evaluation.
4.1 Baselines
Two unsupervised baselines are considered. The "majority vote" baseline determines the answer to a
question based on the most frequent prediction among the forecasts. The "weighted vote" baseline,
on the other hand, assigns weights to the probabilities in the predictions and then aggregates them.
4.2 Neural Network Architecture
A neural network architecture is employed, which consists of three main components: one to generate
a representation of the question, another to generate a representation of each forecast, and an LSTM
to process the sequence of forecasts and ultimately call the question.
The representation of a question is obtained using BERT, followed by a fully connected layer with 256
neurons, ReLU activation, and dropout. The representation of a forecast is created by concatenating
three elements: (a) a binary flag indicating whether the forecast was submitted on the day the question
is being called or on a previous day, (b) the prediction itself (a numerical value between 0.0 and 1.0),
and (c) a representation of the justification. The representation of the justification is also obtained
using BERT, followed by a fully connected layer with 256 neurons, ReLU activation, and dropout.
The LSTM has a hidden state with a dimensionality of 256 and processes the sequence of forecasts
as its input. During the tuning process, it was discovered that providing the representation of the
question alongside each forecast is more effective than processing forecasts independently of the
question. Consequently, the representation of the question is concatenated with the representation of
each forecast before being fed into the LSTM. Finally, the last hidden state of the LSTM is connected
to a fully connected layer with a single neuron and sigmoid activation to produce the final prediction
for the question.
4.3 Architecture Ablation
Experiments are carried out with the complete neural architecture, as described above, as well as
with variations where certain components are disabled. Specifically, the representation of a forecast
is manipulated by incorporating different combinations of information:
4
* Only the prediction. * The prediction and the representation of the question. * The prediction and
the representation of the justification. * The prediction, the representation of the question, and the
representation of the justification.
4.4 Quantitative Results
The evaluation metric used is accuracy, which represents the average percentage of days a model
correctly calls a question throughout its duration. Results are reported for all days combined, as well
as for each of the four quartiles of the question’s duration.
Table 3: Results with the test questions (Accuracy: average percentage of days a model predicts a
question correctly). Results are provided for all days a question was open and for four quartiles (Q1:
first 25% of days, Q2: 25-50%, Q3: 50-75%, and Q4: last 25% of days).
Days When the Question Was Open
Model All Days Q1 Q2 Q3 Q4
Using Daily Forecasts Only
Baselines
Majority Vote (predictions) 71.89 64.59 66.59 73.26 82.22
Weighted Vote (predictions) 73.79 67.79 68.71 74.16 83.61
Neural Network Variants
Predictions Only 77.96 77.62 77.93 78.23 78.61
Predictions + Question 77.61 75.44 76.77 78.05 81.56
Predictions + Justifications 80.23 77.87 78.65 79.26 84.67
Predictions + Question + Justifications 79.96 78.65 78.11 80.29 83.28
Using Active Forecasts
Baselines
Majority Vote (predictions) 77.27 68.83 73.92 77.98 87.44
Weighted Vote (predictions) 77.97 72.04 72.17 78.53 88.22
Neural Network Variants
Predictions Only 78.81 77.31 78.04 78.53 81.11
Predictions + Question 79.35 76.05 78.53 79.56 82.94
Predictions + Justifications 80.84 77.86 79.07 79.74 86.17
Predictions + Question + Justifications 81.27 78.71 79.81 81.56 84.67
Despite their relative simplicity, the baseline methods achieve commendable results, demonstrating
that aggregating forecaster predictions without considering the question or justifications is a viable
strategy. However, the full neural network achieves significantly improved results.
**Using Daily or Active Forecasts** Incorporating active forecasts, rather than solely relying on
forecasts submitted on the day the question is called, proves advantageous for both baselines and all
neural network configurations, except for the one using only predictions and justifications.
**Encoding Questions and Justifications** The neural network that only utilizes the prediction
to represent a forecast surpasses both baseline methods. Notably, integrating the question, the
justification, or both into the forecast representation yields further improvements. These results
indicate that incorporating the question and forecaster-provided justifications into the model enhances
the accuracy of question calling.
**Calling Questions Throughout Their Life** When examining the results across the four quartiles of
a question’s duration, it’s observed that while using active forecasts is beneficial across all quartiles
for both baselines and all network configurations, the neural networks surprisingly outperform the
baselines only in the first three quartiles. In the last quartile, the neural networks perform significantly
worse than the baselines. This suggests that while modeling questions and justifications is generally
helpful, it becomes detrimental toward the end of a question’s life. This phenomenon can be attributed
to the increasing wisdom of the crowd as more evidence becomes available and more forecasters
contribute, making their aggregated predictions more accurate.
5
Table 4: Results with the test questions, categorized by question difficulty as determined by the best
baseline model. The table presents the accuracy (average percentage of days a question is predicted
correctly) for all questions and for each quartile of difficulty: Q1 (easiest 25%), Q2 (25-50%), Q3
(50-75%), and Q4 (hardest 25%).
Question Difficulty (Based on Best Baseline)
All Q1 Q2 Q3 Q4
Using Active Forecasts
Weighted Vote Baseline (Predictions) 77.97 99.40 99.55 86.01 29.30
Neural Network with Components...
Predictions + Question 79.35 94.58 88.01 78.04 58.73
Predictions + Justifications 80.84 95.71 93.18 79.99 57.05
Predictions + Question + Justifications 81.27 94.17 90.11 78.67 64.41
**Calling Questions Based on Their Difficulty** The analysis is further refined by examining
results based on question difficulty, determined by the number of days the best-performing baseline
incorrectly calls the question. This helps to understand which questions benefit most from the neural
networks that incorporate questions and justifications. However, it’s important to note that calculating
question difficulty during the question’s active period is not feasible, making these experiments
unrealistic before the question closes and the correct answer is revealed.
Table 4 presents the results for selected models based on question difficulty. The weighted vote
baseline demonstrates superior performance for 75
5 Qualitative Analysis
This section provides insights into the factors that make questions more difficult to forecast and
examines the characteristics of justifications associated with incorrect and correct predictions.
**Questions** An analysis of the 88 questions in the test set revealed that questions called incorrectly
on at least one day by the best model tend to have a shorter duration (69.4 days vs. 81.7 days) and a
higher number of active forecasts per day (31.0 vs. 26.7). This suggests that the model’s errors align
with the questions that forecasters also find challenging.
**Justifications** A manual review of 400 justifications (200 associated with incorrect predictions
and 200 with correct predictions) was conducted, focusing on those submitted on days when the best
model made an incorrect prediction. The following observations were made:
* A higher percentage of incorrect predictions (78%) were accompanied by short justifications
(fewer than 20 tokens), compared to 65% for correct predictions. This supports the idea that longer
user-generated text often indicates higher quality. * References to previous forecasts (either by the
same or other forecasters, or the current crowd’s forecast) were more common in justifications for
incorrect predictions (31.5%) than for correct predictions (16%). * A lack of a logical argument
was prevalent in the justifications, regardless of the prediction’s accuracy. However, it was more
frequent in justifications for incorrect predictions (62.5%) than for correct predictions (47.5%). *
Surprisingly, justifications with generic arguments did not clearly differentiate between incorrect and
correct predictions (16.0% vs. 14.5%). * Poor grammar and spelling or the use of non-English were
infrequent but more common in justifications for incorrect predictions (24.5%) compared to correct
predictions (14.5%).
6 Conclusions
Forecasting involves predicting future events, a capability highly valued by both governments and
industries as it enables them to anticipate and address potential challenges. This study focuses on
questions spanning the political, economic, and social domains, utilizing forecasts submitted by a
crowd of individuals without specialized training. Each forecast comprises a prediction and a natural
language justification.
6
The research demonstrates that aggregating the weighted predictions of forecasters is a solid baseline
for calling a question throughout its duration. However, models that incorporate both the question
and the justifications achieve significantly better results, particularly during the first three quartiles of
a question’s life. Importantly, the models developed in this study do not profile individual forecasters
or utilize any information about their identities. This work lays the groundwork for evaluating the
credibility of anonymous forecasts, enabling the development of robust aggregation strategies that do
not require tracking individual forecasters.
"""
content_r011= """
Addressing Popularity Bias with Popularity-Conscious Alignment and
Contrastive Learning
Abstract
Collaborative Filtering (CF) often encounters substantial difficulties with popularity bias because of the skewed
distribution of items in real-world datasets. This tendency creates a notable difference in accuracy between items
that are popular and those that are not. This discrepancy impedes the accurate comprehension of user preferences
and intensifies the Matthew effect within recommendation systems. To counter popularity bias, current methods
concentrate on highlighting less popular items or on differentiating the correlation between item representations
and their popularity. Despite their effectiveness, current approaches continue to grapple with two significant
issues: firstly, the extraction of shared supervisory signals from popular items to enhance the representations of
less popular items, and secondly, the reduction of representation separation caused by popularity bias. In this
study, we present an empirical examination of popularity bias and introduce a method called Popularity-Aware
Alignment and Contrast (PAAC) to tackle these two problems. Specifically, we utilize the common supervisory
signals found in popular item representations and introduce an innovative popularity-aware supervised alignment
module to improve the learning of representations for unpopular items. Furthermore, we propose adjusting the
weights in the contrastive learning loss to decrease the separation of representations by focusing on popularity.
We confirm the efficacy and logic of PAAC in reducing popularity bias through thorough experiments on three
real-world datasets.
1 Introduction
Contemporary recommender systems are essential in reducing information overload. Personalized recommendations frequently
employ collaborative filtering (CF) to assist users in discovering items that may interest them. CF-based techniques primarily
learn user preferences and item attributes by matching the representations of users with the items they engage with. Despite their
achievements, CF-based methods frequently encounter the issue of popularity bias, which leads to considerable disparities in
accuracy between items that are popular and those that are not. Popularity bias occurs because there are limited supervisory signals
for items that are not popular, which results in overfitting during the training phase and decreased effectiveness on the test set. This
hinders the precise comprehension of user preferences, thereby diminishing the variety of recommendations. Furthermore, popularity
bias can worsen the Matthew effect, where items that are already popular gain even more popularity because they are recommended
more frequently.
Two significant challenges are presented when mitigating popularity bias in recommendation systems. The first challenge is the
inadequate representation of unpopular items during training, which results in overfitting and limited generalization ability. The
second challenge, known as representation separation, happens when popular and unpopular items are categorized into distinct
semantic spaces, thereby intensifying the bias and diminishing the precision of recommendations.
2 Methodology
To overcome the current difficulties in reducing popularity bias, we introduce the Popularity-Aware Alignment and Contrast (PAAC)
method. We utilize the common supervisory signals present in popular item representations to direct the learning of unpopular
representations, and we present a popularity-aware supervised alignment module. Moreover, we incorporate a re-weighting system
in the contrastive learning module to deal with representation separation by considering popularity.
2.1 Supervised Alignment Module
During the training process, the alignment of representations usually emphasizes users and items that have interacted, often causing
items to be closer to interacted users than non-interacted ones in the representation space. However, because unpopular items have
limited interactions, they are usually modeled based on a small group of users. This limited focus can result in overfitting, as the
representations of unpopular items might not fully capture their features.
The disparity in the quantity of supervisory signals is essential for learning representations of both popular and unpopular items.
Specifically, popular items gain from a wealth of supervisory signals during the alignment process, which helps in effectively
learning their representations. On the other hand, unpopular items, which have a limited number of users providing supervision, are
more susceptible to overfitting. This is because there is insufficient representation learning for unpopular items, emphasizing the
effect of supervisory signal distribution on the quality of representation. Intuitively, items interacted with by the same user have
some similar characteristics. In this section, we utilize common supervisory signals in popular item representations and suggest a
popularity-aware supervised alignment method to improve the representations of unpopular items.
We initially filter items with similar characteristics based on the user’s interests. For any user, we define the set of items they interact
with. We count the frequency of each item appearing in the training dataset as its popularity. Subsequently, we group items based on
their relative popularity. We divide items into two groups: the popular item group and the unpopular item group. The popularity of
each item in the popular group is higher than that of any item in the unpopular group. This indicates that popular items receive more
supervisory information than unpopular items, resulting in poorer recommendation performance for unpopular items.
To tackle the issue of insufficient representation learning for unpopular items, we utilize the concept that items interacted with by the
same user share some similar characteristics. Specifically, we use similar supervisory signals in popular item representations to
improve the representations of unpopular items. We align the representations of items to provide more supervisory information to
unpopular items and improve their representation, as follows:
LSA = X
u∈U
1
|Iu|
X
i∈Iu
pop,j∈Iu
unpop
||f (i) − f (j)||2, (1)
where f (·) is a recommendation encoder and hi = f (i). By efficiently using the inherent information in the data, we provide more
supervisory signals for unpopular items without introducing additional side information. This module enhances the representation of
unpopular items, mitigating the overfitting issue.
2.2 Re-weighting Contrast Module
Recent research has indicated that popularity bias frequently leads to a noticeable separation in the representation of item embeddings.
Although methods based on contrastive learning aim to enhance overall uniformity by distancing negative samples, their current
sampling methods might unintentionally worsen this separation. When negative samples follow the popularity distribution, which
is dominated by popular items, prioritizing unpopular items as positive samples widens the gap between popular and unpopular
items in the representation space. Conversely, when negative samples follow a uniform distribution, focusing on popular items
separates them from most unpopular ones, thus worsening the representation gap. Existing studies use the same weights for positive
and negative samples in the contrastive loss function, without considering differences in item popularity. However, in real-world
recommendation datasets, the impact of items varies due to dataset characteristics and interaction distributions. Neglecting this
aspect could lead to suboptimal results and exacerbate representation separation.
We propose to identify different influences by re-weighting different popularity items. To this end, we introduce re-weighting
different positive and negative samples to mitigate representation separation from a popularity-centric perspective. We incorporate
this approach into contrastive learning to better optimize the consistency of representations. Specifically, we aim to reduce the risk
of pushing items with varying popularity further apart. For example, when using a popular item as a positive sample, our goal is
to avoid pushing unpopular items too far away. Thus, we introduce two hyperparameters to control the weights when items are
considered positive and negative samples.
To ensure balanced and equitable representations of items within our model, we first propose a dynamic strategy to categorize items
into popular and unpopular groups for each mini-batch. Instead of relying on a fixed global threshold, which often leads to the
overrepresentation of popular items across various batches, we implement a hyperparameter x. This hyperparameter readjusts the
classification of items within the current batch. By adjusting the hyperparameter x, we maintain a balance between different item
popularity levels. This enhances the model’s ability to generalize across diverse item sets by accurately reflecting the popularity
distribution in the current training context. Specifically, we denote the set of items within each batch as IB . And then we divide IB
into a popular group Ipop and an unpopular group Iunpop based on their respective popularity levels, classifying the top x% of items
as Ipop:
IB = Ipop ∪ Iunpop, ∀i ∈ Ipop ∧ j ∈ Iunpop, p(i) > p(j), (2)
where Ipop ∈ IB and Iunpop ∈ IB are disjoint, with Ipop consisting of the top x% of items in the batch. In this work, we dynamically
divided items into popular and unpopular groups within each mini-batch based on their popularity, assigning the top 50% as popular
items and the bottom 50% as unpopular items. This radio not only ensures equal representation of both groups in our contrastive
learning but also allows items to be classified adaptively based on the batch’s current composition.
After that, we use InfoNCE to optimize the uniformity of item representations. Unlike traditional CL-based methods, we calculate
the loss for different item groups. Specifically, we introduce the hyperparameter α to control the positive sample weights between
popular and unpopular items, adapting to varying item distributions in different datasets:
2
LCL
item = α × LCL
pop + (1 − α) × LCL
unpop, (3)
where LCL
pop represents the contrastive loss when popular items are considered as positive samples, and LCL
unpop represents the
contrastive loss when unpopular items are considered as positive samples. The value of α ranges from 0 to 1, where α = 0 means
exclusive emphasis on the loss of unpopular items LCL
unpop, and α = 1 means exclusive emphasis on the loss of popular items
LCL
pop. By adjusting α, we can effectively balance the impact of positive samples from both popular and unpopular items, allowing
adaptability to varying item distributions in different datasets.
Following this, we fine-tune the weighting of negative samples in the contrastive learning framework using the hyperparameter β.
This parameter controls how samples from different popularity groups contribute as negative samples. Specifically, we prioritize
re-weighting items with popularity opposite to the positive samples, mitigating the risk of excessively pushing negative samples
away and reducing representation separation. Simultaneously, this approach ensures the optimization of intra-group consistency. For
instance, when dealing with popular items as positive samples, we separately calculate the impact of popular and unpopular items
as negative samples. The hyperparameter β is then used to control the degree to which unpopular items are pushed away. This is
formalized as follows:
L′
pop = X
i∈Ipop
log exp(h′
ihi/τ )
P
j∈Ipop exp(h′
ihj /τ ) + β P
j∈Iunpop exp(h′
ihj /τ ) , (4)
similarly, the contrastive loss for unpopular items is defined as:
L′
unpop = X
i∈Iunpop
log exp(h′
ihi/τ )
P
j∈Iunpop exp(h′
ihj /τ ) + β P
j∈Ipop exp(h′
ihj /τ ) , (5)
where the parameter β ranges from 0 to 1, controlling the negative sample weighting in the contrastive loss. When β = 0, it means
that only intra-group uniformity optimization is performed. Conversely, when β = 1, it means equal treatment of both popular and
unpopular items in terms of their impact on positive samples. The setting of β allows for a flexible adjustment between prioritizing
intra-group uniformity and considering the impact of different popularity levels in the training. We prefer to push away items
within the same group to optimize uniformity. This setup helps prevent over-optimizing the uniformity of different groups, thereby
mitigating representation separation.
The final re-weighting contrastive objective is the weighted sum of the user objective and the item objective:
LCL = 1
2 × (LCL
item + LCL
user ). (6)
In this way, we not only achieved consistency in representation but also reduced the risk of further separating items with similar
characteristics into different representation spaces, thereby alleviating the issue of representation separation caused by popularity
bias.
2.3 Model Optimization
To reduce popularity bias in collaborative filtering tasks, we employ a multi-task training strategy to jointly optimize the classic
recommendation loss (LREC ), supervised alignment loss (LSA), and re-weighting contrast loss (LCL).
L = LREC + λ1LSA + λ2LCL + λ3||Θ||2, (7)
where Θ is the set of model parameters in LREC as we do not introduce additional parameters, λ1 and λ2 are hyperparameters that
control the strengths of the popularity-aware supervised alignment loss and the re-weighting contrastive learning loss respectively,
and λ3 is the L2 regularization coefficient. After completing the model training process, we use the dot product to predict unknown
preferences for recommendations.
3 Experiments
In this section, we assess the efficacy of PAAC through comprehensive experiments, aiming to address the following research
questions:
• How does PAAC compare to existing debiasing methods?
• How do different designed components play roles in our proposed PAAC?
3
• How does PAAC alleviate the popularity bias?
• How do different hyper-parameters affect the PAAC recommendation performance?
3.1 Experiments Settings
3.1.1 Datasets
In our experiments, we use three widely public datasets: Amazon-book, Yelp2018, and Gowalla. We retained users and items with a
minimum of 10 interactions.
3.1.2 Baselines and Evaluation Metrics
We implement the state-of-the-art LightGCN to instantiate PAAC, aiming to investigate how it alleviates popularity bias. We
compare PAAC with several debiased baselines, including re-weighting-based models, decorrelation-based models, and contrastive
learning-based models.
We utilize three widely used metrics, namely Recall@K, HR@K, and NDCG@K, to evaluate the performance of Top-K recommen-
dation. Recall@K and HR@K assess the number of target items retrieved in the recommendation results, emphasizing coverage. In
contrast, NDCG@K evaluates the positions of target items in the ranking list, with a focus on their positions in the list. We use
the full ranking strategy, considering all non-interacted items as candidate items to avoid selection bias during the test stage. We
repeated each experiment five times with different random seeds and reported the average scores.
3.2 Overall Performance
As shown in Table 1, we compare our model with several baselines across three datasets. The best performance for each metric
is highlighted in bold, while the second best is underlined. Our model consistently outperforms all compared methods across all
metrics in every dataset.
• Our proposed model PAAC consistently outperforms all baselines and significantly mitigates the popularity bias. Specif-
ically, PAAC enhances LightGCN, achieving improvements of 282.65%, 180.79%, and 82.89% in NDCG@20 on the
Yelp2018, Gowalla, and Amazon-Book datasets, respectively. Compared to the strongest baselines, PAAC delivers better
performance. The most significant improvements are observed on Yelp2018, where our model achieves an 8.70% increase
in Recall@20, a 10.81% increase in HR@20, and a 30.2% increase in NDCG@20. This improvement can be attributed
to our use of popularity-aware supervised alignment to enhance the representation of less popular items and re-weighted
contrastive learning to address representation separation from a popularity-centric perspective.
• The performance improvements of PAAC are smaller on sparser datasets. For example, on the Gowalla dataset, the
improvements in Recall@20, HR@20, and NDCG@20 are 3.18%, 5.85%, and 5.47%, respectively. This may be because,
in sparser datasets like Gowalla, even popular items are not well-represented due to lower data density. Aligning unpopular
items with these poorly represented popular items can introduce noise into the model. Therefore, the benefits of using
supervisory signals for unpopular items may be reduced in very sparse environments, leading to smaller performance
improvements.
• Regarding the baselines for mitigating popularity bias, the improvement of some is relatively limited compared to the
backbone model (LightGCN) and even performs worse in some cases. This may be because some are specifically designed
for traditional data-splitting scenarios, where the test set still follows a long-tail distribution, leading to poor generalization.
Some mitigate popularity bias by excluding item popularity information. Others use invariant learning to remove popularity
information at the representation level, generally performing better than the formers. This shows the importance of
addressing popularity bias at the representation level. Some outperform the other baselines, emphasizing the necessary to
improve item representation consistency for mitigating popularity bias.
• Different metrics across various datasets show varying improvements in model performance. This suggests that different
debiasing methods may need distinct optimization strategies for models. Additionally, we observe varying effects of PAAC
across different datasets. This difference could be due to the sparser nature of the Gowalla dataset. Conversely, our model
can directly provide supervisory signals for unpopular items and conduct intra-group optimization, consistently maintaining
optimal performance across all metrics on the three datasets.
3.3 Ablation Study
To better understand the effectiveness of each component in PAAC, we conduct ablation studies on three datasets. Table 2 presents a
comparison between PAAC and its variants on recommendation performance. Specifically, PAAC-w/o P refers to the variant where
the re-weighting contrastive loss of popular items is removed, focusing instead on optimizing the consistency of representations for
unpopular items. Similarly, PAAC-w/o U denotes the removal of the re-weighting contrastive loss for unpopular items. PAAC-w/o
A refers to the variant without the popularity-aware supervised alignment loss. It’s worth noting that PAAC-w/o A differs from
4
Table 1: Performance comparison on three public datasets with K = 20. The best performance is indicated in bold, while the
second-best performance is underlined. The superscripts * indicate p ≤ 0.05 for the paired t-test of PAAC vs. the best baseline (the
relative improvements are denoted as Imp.).
!
Model Yelp2018 Gowalla Amazon-book
Recall@20 HR@20 NDCG@20 Recall@20 HR@20 NDCG@20 Recall@20 HR@20 NDCG@20
MF 0.0050 0.0109 0.0093 0.0343 0.0422 0.0280 0.0370 0.0388 0.0270
LightGCN 0.0048 0.0111 0.0098 0.0380 0.0468 0.0302 0.0421 0.0439 0.0304
IPS 0.0104 0.0183 0.0158 0.0562 0.0670 0.0444 0.0488 0.0510 0.0365
MACR 0.0402 0.0312 0.0265 0.0908 0.1086 0.0600 0.0515 0.0609 0.0487
α-Adjnorm 0.0053 0.0088 0.0080 0.0328 0.0409 0.0267 0.0422 0.0450 0.0264
InvCF 0.0444 0.0344 0.0291 0.1001 0.1202 0.0662 0.0562 0.0665 0.0515
Adap-τ 0.0450 0.0497 0.0341 0.1182 0.1248 0.0794 0.0641 0.0678 0.0511
SimGCL 0.0449 0.0518 0.0345 0.1194 0.1228 0.0804 0.0628 0.0648 0.0525
PAAC 0.0494* 0.0574* 0.0375* 0.1232* 0.1321* 0.0848* 0.0701* 0.0724* 0.0556*
Imp. +9.78 % +10.81% +8.70% +3.18% +5.85% +5.47% +9.36% +6.78% 5.90%
SimGCL in that we split the contrastive loss on the item side, LCL
item, into two distinct losses: LCL
pop and LCL
unpop. This approach
allows us to separately address the consistency of popular and unpopular item representations, thereby providing a more detailed
analysis of the impact of each component on the overall performance.
From Table 2, we observe that PAAC-w/o A outperforms SimGCL in most cases. This validates that re-weighting the importance of
popular and unpopular items can effectively improve the model’s performance in alleviating popularity bias. It also demonstrates the
effectiveness of using supervision signals from popular items to enhance the representations of unpopular items, providing more
opportunities for future research on mitigating popularity bias. Moreover, compared with PAAC-w/o U, PAAC-w/o P results in much
worse performance. This confirms the importance of re-weighting popular items in contrastive learning for mitigating popularity
bias. Finally, PAAC consistently outperforms the three variants, demonstrating the effectiveness of combining supervised alignment
and re-weighting contrastive learning. Based on the above analysis, we conclude that leveraging supervisory signals from popular
item representations can better optimize representations for unpopular items, and re-weighting contrastive learning allows the model
to focus on more informative or critical samples, thereby improving overall performance. All the proposed modules significantly
contribute to alleviating popularity bias.
Table 2: Ablation study of PAAC, highlighting the best-performing model on each dataset and metrics in bold. Specifically,
PAAC-w/o P removes the re-weighting contrastive loss of popular items, PAAC-w/o U eliminates the re-weighting contrastive loss
of unpopular items, and PAAC-w/o A omits the popularity-aware supervised alignment loss.
!
Model Yelp2018 Gowalla Amazon-book
Recall@20 HR@20 NDCG@20 Recall@20 HR@20 NDCG@20 Recall@20 HR@20 NDCG@20
SimGCL 0.0449 0.0518 0.0345 0.1194 0.1228 0.0804 0.0628 0.0648 0.0525
PAAC-w/o P 0.0443 0.0536 0.0340 0.1098 0.1191 0.0750 0.0616 0.0639 0.0458
PAAC-w/o U 0.0462 0.0545 0.0358 0.1120 0.1179 0.0752 0.0594 0.0617 0.0464
PAAC-w/o A 0.0466 0.0547 0.0360 0.1195 0.1260 0.0815 0.0687 0.0711 0.0536
PAAC 0.0494* 0.0574* 0.0375* 0.1232* 0.1321* 0.0848* 0.0701* 0.0724* 0.0556*
3.4 Debias Ability
To further verify the effectiveness of PAAC in alleviating popularity bias, we conduct a comprehensive analysis focusing on the
recommendation performance across different popularity item groups. Specifically, 20% of the most popular items are labeled
’Popular’, and the rest are labeled ’Unpopular’. We compare the performance of PAAC with LightGCN, IPS, MACR, and SimGCL
using the NDCG@20 metric across different popularity groups. We use ∆ to denote the accuracy gap between the two groups. We
draw the following conclusions:
• Improving the performance of unpopular items is crucial for enhancing overall model performance. Specially, on the
Yelp2018 dataset, PAAC shows reduced accuracy in recommending popular items, with a notable decrease of 20.14%
compared to SimGCL. However, despite this decrease, the overall recommendation accuracy surpasses that of SimGCL
by 11.94%, primarily due to a 6.81% improvement in recommending unpopular items. This improvement highlights the
importance of better recommendations for unpopular items and emphasizes their crucial role in enhancing overall model
performance.
5
• Our proposed PAAC significantly enhances the recommendation performance for unpopular items. Specifically, we observe
an improvement of 8.94% and 7.30% in NDCG@20 relative to SimGCL on the Gowalla and Yelp2018 datasets, respectively.
This improvement is due to the popularity-aware alignment method, which uses supervisory signals from popular items to
improve the representations of unpopular items.
• PAAC has successfully narrowed the accuracy gap between different item groups. Specifically, PAAC achieved the smallest
gap, reducing the NDCG@20 accuracy gap by 34.18% and 87.50% on the Gowalla and Yelp2018 datasets, respectively.
This indicates that our method treats items from different groups fairly, effectively alleviating the impact of popularity
bias. This success can be attributed to our re-weighted contrast module, which addresses representation separation from a
popularity-centric perspective, resulting in more consistent recommendation results across different groups.
3.5 Hyperparameter Sensitivities
In this section, we analyze the impact of hyperparameters in PAAC. Firstly, we investigate the influence of λ1 and λ2, which
respectively control the impact of the popularity-aware supervised alignment and re-weighting contrast loss. Additionally, in the
re-weighting contrastive loss, we introduce two hyperparameters, α and β, to control the re-weighting of different popularity items
as positive and negative samples. Finally, we explore the impact of the grouping ratio x on the model’s performance.
3.5.1 Effect of λ1 and λ2
As formulated in Eq. (11), λ1 controls the extent of providing additional supervisory signals for unpopular items, while λ2 controls
the extent of optimizing representation consistency. Horizontally, with the increase in λ2, the performance initially increases and
then decreases. This indicates that appropriate re-weighting contrastive loss effectively enhances the consistency of representation
distributions, mitigating popularity bias. However, overly strong contrastive loss may lead the model to neglect recommendation
accuracy. Vertically, as λ1 increases, the performance also initially increases and then decreases. This suggests that suitable
alignment can provide beneficial supervisory signals for unpopular items, while too strong an alignment may introduce more noise
from popular items to unpopular ones, thereby impacting recommendation performance.
3.5.2 Effect of re-weighting coefficient α and β
To mitigate representation separation due to imbalanced positive and negative sampling, we introduce two hyperparameters into the
contrastive loss. Specifically, α controls the weight difference between positive samples from popular and unpopular items, while β
controls the influence of different popularity items as negative samples.
In our experiments, while keeping other hyperparameters constant, we search α and β within the range {0, 0.2, 0.4, 0.6, 0.8, 1}. As
α and β increase, performance initially improves and then declines. The optimal hyperparameters for the Yelp2018 and Gowalla
datasets are α = 0.8, β = 0.6 and α = 0.2, β = 0.2, respectively. This may be attributed to the characteristics of the datasets. The
Yelp2018 dataset, with a higher average interaction frequency per item, benefits more from a higher weight α for popular items as
positive samples. Conversely, the Gowalla dataset, being relatively sparse, prefers a smaller α. This indicates the importance of
considering dataset characteristics when adjusting the contributions of popular and unpopular items to the model.
Notably, α and β are not highly sensitive within the range [0, 1], performing well across a broad spectrum. Performance exceeds the
baseline regardless of β values when other parameters are optimal. Additionally, α values from [0.4, 1.0] on the Yelp2018 dataset
and [0.2, 0.8] on the Gowalla dataset surpass the baseline, indicating less need for precise tuning. Thus, α and β achieve optimal
performance without meticulous adjustments, focusing on weight coefficients to maintain model efficacy.
3.5.3 Effect of grouping ratio x
To investigate the impact of different grouping ratios on recommendation performance, we developed a flexible classification
method for items within each mini-batch based on their popularity. Instead of adopting a fixed global threshold, which tends to
overrepresent popular items in some mini-batches, our approach dynamically divides items in each mini-batch into popular and
unpopular categories. Specifically, the top x% of items are classified as popular and the remaining (100 - x)% as unpopular, with x
varying. This strategy prevents the overrepresentation typical in fixed distribution models, which could skew the learning process
and degrade performance. To quantify the effects of these varying ratios, we examined various division ratios for popular items,
including 20%, 40%, 60%, and 80%, as shown in Table 3. The preliminary results indicate that both extremely low and high ratios
negatively affect model performance, thereby underscoring the superiority of our dynamic data partitioning approach. Moreover,
within the 40%-60% range, our model’s performance remained consistently robust, further validating the effectiveness of PAAC.
6
Table 3: Performance comparison across varying popular item ratios x on metrics.
!
Ratio Yelp2018 Gowalla
Recall@20 HR@20 NDCG@20 Recall@20 HR@20 NDCG@20
20% 0.0467 0.0555 0.0361 0.1232 0.1319 0.0845
40% 0.0505 0.0581 0.0378 0.1239 0.1325 0.0848
50% 0.0494 0.0574 0.0375 0.1232 0.1321 0.0848
60% 0.0492 0.0569 0.0370 0.1225 0.1314 0.0843
80% 0.0467 0.0545 0.0350 0.1176 0.1270 0.0818
4 Related Work
4.1 Popularity Bias in Recommendation
Popularity bias is a prevalent problem in recommender systems, where unpopular items in the training dataset are seldom recom-
mended. Numerous techniques have been suggested to examine and decrease performance variations between popular and unpopular
items. These techniques can be broadly divided into three categories.
• Re-weighting-based methods aim to increase the training weight or scores for unpopular items, redirecting focus away
from popular items during training or prediction. For instance, IPS adds compensation to unpopular items and adjusts
the prediction of the user-item preference matrix, resulting in higher preference scores and improving rankings for
unpopular items. α-AdjNorm enhances the focus on unpopular items by controlling the normalization strength during the
neighborhood aggregation process in GCN-based models.
• Decorrelation-based methods aim to effectively remove the correlations between item representations (or prediction scores)
and popularity. For instance, MACR uses counterfactual reasoning to eliminate the direct impact of popularity on item
outcomes. In contrast, InvCF operates on the principle that item representations remain invariant to changes in popularity
semantics, filtering out unstable or outdated popularity characteristics to learn unbiased representations.
• Contrastive-learning-based methods aim to achieve overall uniformity in item representations using InfoNCE, preserving
more inherent characteristics of items to mitigate popularity bias. This approach has been demonstrated as a state-of-the-art
method for alleviating popularity bias. It employs data augmentation techniques such as graph augmentation or feature
augmentation to generate different views, maximizing positive pair consistency and minimizing negative pair consistency
to promote more uniform representations. Specifically, Adap-τ adjusts user/item embeddings to specific values, while
SimGCL integrates InfoNCE loss to enhance representation uniformity and alleviate popularity bias.
4.2 Representation Learning for CF
Representation learning is crucial in recommendation systems, especially in modern collaborative filtering (CF) techniques. It
creates personalized embeddings that capture user preferences and item characteristics. The quality of these representations critically
determines a recommender system’s effectiveness by precisely capturing the interplay between user interests and item features.
Recent studies emphasize two fundamental principles in representation learning: alignment and uniformity. The alignment principle
ensures that embeddings of similar or related items (or users) are closely clustered together, improving the system’s ability to
recommend items that align with a user’s interests. This principle is crucial when accurately reflecting user preferences through
corresponding item characteristics. Conversely, the uniformity principle ensures a balanced distribution of all embeddings across the
representation space. This approach prevents the over-concentration of embeddings in specific areas, enhancing recommendation
diversity and improving generalization to unseen data.
In this work, we focus on aligning the representations of popular and unpopular items interacted with by the same user and re-
weighting uniformity to mitigate representation separation. Our model PAAC uniquely addresses popularity bias by combining group
alignment and contrastive learning, a first in the field. Unlike previous works that align positive user-item pairs or contrastive pairs,
PAAC directly aligns popular and unpopular items, leveraging the rich information of popular items to enhance the representations
of unpopular items and reduce overfitting. Additionally, we introduce targeted re-weighting from a popularity-centric perspective to
achieve a more balanced representation.
5 Conclusion
In this study, we have examined popularity bias and put forward PAAC as a method to lessen its impact. We postulated that items
engaged with by the same user exhibit common traits, and we utilized this insight to coordinate the representations of both popular
and unpopular items via a popularity-conscious supervised alignment method. This strategy furnished additional supervisory data for
less popular items. It is important to note that our concept of aligning and categorizing items according to user-specific preferences
introduces a fresh perspective on alignment. Moreover, we tackled the problem of representation separation seen in current CL-based
7
models by incorporating two hyperparameters to regulate the influence of items with varying popularity levels when considered
as positive and negative samples. This method refined the uniformity of representations and successfully reduced separation. We
validated our method, PAAC, on three publicly available datasets, demonstrating its effectiveness and underlying rationale.
In the future, we will explore deeper alignment and contrast adjustments tailored to specific tasks to further mitigate popularity
bias. We aim to investigate the synergies between alignment and contrast and extend our approach to address other biases in
recommendation systems.
Acknowledgments
This work was supported in part by grants from the National Key Research and Development Program of China, the National Natural
Science Foundation of China, the Fundamental Research Funds for the Central Universities, and Quan Cheng Laboratory.

"""
content_r013 = """
Generalization in ReLU Networks via Restricted
Isometry and Norm Concentration
Abstract
Regression tasks, while aiming to model relationships across the entire input space,
are often constrained by limited training data. Nevertheless, if the hypothesis func-
tions can be represented effectively by the data, there is potential for identifying a
model that generalizes well. This paper introduces the Neural Restricted Isometry
Property (NeuRIPs), which acts as a uniform concentration event that ensures all
shallow ReLU networks are sketched with comparable quality. To determine the
sample complexity necessary to achieve NeuRIPs, we bound the covering numbers
of the networks using the Sub-Gaussian metric and apply chaining techniques. As-
suming the NeuRIPs event, we then provide bounds on the expected risk, applicable
to networks within any sublevel set of the empirical risk. Our results show that all
networks with sufficiently small empirical risk achieve uniform generalization.
1 Introduction
A fundamental requirement of any scientific model is a clear evaluation of its limitations. In recent
years, supervised machine learning has seen the development of tools for automated model discovery
from training data. However, these methods often lack a robust theoretical framework to estimate
model limitations. Statistical learning theory quantifies the limitation of a trained model by the
generalization error. This theory uses concepts such as the VC-dimension and Rademacher complexity
to analyze generalization error bounds for classification problems. While these traditional complexity
notions have been successful in classification problems, they do not apply to generic regression
problems with unbounded risk functions, which are the focus of this study. Moreover, traditional
tools in statistical learning theory have not been able to provide a fully satisfying generalization
theory for neural networks.
Understanding the risk surface during neural network training is crucial for establishing a strong
theoretical foundation for neural network-based machine learning, particularly for understanding
generalization. Recent studies on neural networks suggest intriguing properties of the risk surface.
In large networks, local minima of the risk form a small bond at the global minimum. Surprisingly,
global minima exist in each connected component of the risk’s sublevel set and are path-connected.
In this work, we contribute to a generalization theory for shallow ReLU networks, by giving uniform
generalization error bounds within the empirical risk’s sublevel set. We use methods from the analysis
of convex linear regression, where generalization bounds for empirical risk minimizers are derived
from recent advancements in stochastic processes’ chaining theory. Empirical risk minimization
for non-convex hypothesis functions cannot generally be solved efficiently. However, under certain
assumptions, it is still possible to derive generalization error bounds, as we demonstrate in this paper
for shallow ReLU networks. Existing works have applied methods from compressed sensing to
bound generalization errors for arbitrary hypothesis functions. However, they do not capture the
risk’s stochastic nature through the more advanced chaining theory.
This paper is organized as follows. We begin in Section II by outlining our assumptions about the
parameters of shallow ReLU networks and the data distribution to be interpolated. The expected and
empirical risk are introduced in Section III, where we define the Neural Restricted Isometry Property
.
(NeuRIPs) as a uniform norm concentration event. We present a bound on the sample complexity for
achieving NeuRIPs in Theorem 1, which depends on both the network architecture and parameter
assumptions. We provide upper bounds on the generalization error that are uniformly applicable
across the sublevel sets of the empirical risk in Section IV. We prove this property in a network
recovery setting in Theorem 2, and also an agnostic learning setting in Theorem 3. These results
ensure a small generalization error, when any optimization algorithm finds a network with a small
empirical risk. We develop the key proof techniques for deriving the sample complexity of achieving
NeuRIPs in Section V, by using the chaining theory of stochastic processes. The derived results are
summarized in Section VI, where we also explore potential future research directions.
2 Notation and Assumptions
In this section, we will define the key notations and assumptions for the neural networks examined
in this study. A Rectified Linear Unit (ReLU) function ϕ : R → R is given by ϕ(x) := max(x, 0).
Given a weight vector w ∈ Rd, a bias b ∈ R, and a sign κ ∈ {±1}, a ReLU neuron is a function
ϕ(w, b, κ) : Rd → R defined as
ϕ(w, b, κ)(x) = κϕ(wT x + b).
Shallow neural networks are constructed as weighted sums of neurons. Typically they are represented
by a graph with n neurons in a single hidden layer. When using the ReLU activation function, we can
apply a symmetry procedure to represent these as sums:
¯ϕ¯p(x) =
nX
i=0
ϕpi (x),
where ¯p is the tuple (p1, . . . , pn).
Assumption 1. The parameters ¯p, which index shallow ReLU networks, are drawn from a set
¯P ⊆ (Rd × R × {±1})n.
For ¯P , we assume there exist constants cw ≥ 0 and cb ∈ [1, 3], such that for all parameter tuples
¯p = {(w1, b1, κ1), . . . , (wn, bn, κn)} ∈ ¯P , we have
∥wi∥ ≤ cw and |bi| ≤ cb.
We denote the set of shallow networks indexed by a parameter set ¯P by
Φ ¯P := {ϕ¯p : ¯p ∈ ¯P }.
We now equip the input space Rd of the networks with a probability distribution. This distribution
reflects the sampling process and makes each neural network a random variable. Additionally, a
random label y takes its values in the output space R, for which we assume the following.
Assumption 2. The random sample x ∈ Rd and label y ∈ R follow a joint distribution µ such that
the marginal distribution µx of sample x is standard Gaussian with density
1
(2π)d/2 exp

− ∥x∥2
2

.
As available data, we assume independent copies {(xj , yj )}m
j=1 of the random pair (x, y), each
distributed by µ.
3 Concentration of the Empirical Norm
Supervised learning algorithms interpolate labels y for samples x, both distributed jointly by µ on
X × Y. This task is often solved under limited data accessibility. The training data, respecting
Assumption 2, consists of m independent copies of the random pair (x, y). During training, the
interpolation quality of a hypothesis function f : X → Y can only be assessed at the given random
samples {xj }m
j=1. Any algorithm therefore accesses each function f through its sketch samples
S[f ] = (f (x1), . . . , f (xm)),
2
where S is the sample operator. After training, the quality of a resulting model is often measured by
its generalization to new data not used during training. With Rd × R as the input and output space,
we quantify a function f ’s generalization error with its expected risk:
Eµ[f ] := Eµ|y − f (x)|2.
The functional || · ||µ, also gives the norm of the space L2(Rd, µx), which consists of functions
f : Rd → R with
∥f ∥2
µ := Eµx [|f (x)|2].
If the label y depends deterministically on the associated sample x, we can treat y as an element of
L2(Rd, µx), and the expected risk of any function f is the function’s distance to y. By sketching any
hypothesis function f with the sample operator S, we perform a Monte-Carlo approximation of the
expected risk, which is termed the empirical risk:
∥f ∥2
m := 1
m
mX
j=1
(f (xj ) − yj )2 = 1
√m (y1, . . . , ym)T − S[f ]
2
2
.
The random functional || · ||m also defines a seminorm on L2(Rd, µx), referred to as the empirical
norm. Under mild assumptions, || · ||m fails to be a norm.
In order to obtain a well generalizing model, the goal is to identify a function f with a low expected
risk. However, with limited data, we are restricted to optimizing the empirical risk. Our strategy for
deriving generalization guarantees is based on the stochastic relation between both risks. If {xj }m
j=1
are independently distributed by µx, the law of large numbers implies that for any f ∈ L2(Rd, µx)
the convergence
lim
m→∞ ∥f ∥m = ∥f ∥µ.
While this establishes the asymptotic convergence of the empirical norm to the function norm for a
single function f , we have to consider two issues to formulate our concept of norm concentration:
First, we need non-asymptotic results, that is bounds on the distance |∥f ∥m − ∥f ∥µ| for a fixed
number of samples m. Second, the bounds on the distance need to be uniformly valid for all functions
f in a given set.
Sample operators which have uniform concentration properties have been studied as restricted
isometries in the area of compressed sensing. For shallow ReLU networks of the form (1), we define
the restricted isometry property of the sampling operator S as follows.
Definition 1. Let s ∈ (0, 1) be a constant and ¯P be a parameter set. We say that the Neural Restricted
Isometry Property (NeuRIPs( ¯P )) is satisfied if, for all ¯p ∈ ¯P it holds that
(1 − s)∥ϕ¯p∥µ ≤ ∥ϕ¯p∥m ≤ (1 + s)∥ϕ¯p∥µ.
In the following Theorem, we provide a bound on the number m of samples, which is sufficient for
the operator S to satisfy NeuRIPs( ¯P ).
Theorem 1. There exist universal constants C1, C2 ∈ R such that the following holds: For
any sample operator S, constructed from random samples {xj }, respecting Assumption 2, let
¯P ⊂ (Rd × R × {±1})n be any parameter set satisfying Assumption 1 and ||ϕ¯p||µ > 1 for all
¯p ∈ ¯P . Then, for any u > 2 and s ∈ (0, 1), NeuRIPs( ¯P ) is satisfied with probability at least
1 − 17 exp(−u/4) provided that
m ≥ n3c2
w
(1 − s)2 max

C1
(8cb + d + ln(2))
u , C2
n2c2
w
(u/s)2

.
One should notice that, in Theorem 1, there is a tradeoff between the parameter s, which limits the
deviation | ∥ · ∥m − ∥ · ∥µ|, and the confidence parameter u. The lower bound on the corresponding
sample size m is split into two scaling regimes when understanding the quotient u of |∥·∥m −∥·∥µ|/s
as a precision parameter. While in the regime of low deviations and high probabilities the sample size
m must scale quadratically with u/s, in the regime of less precise statements one observes a linear
scaling.
3
4 Uniform Generalization of Sublevel Sets of the Empirical Risk
When the NeuRIPs event occurs, the function norm || · ||µ, which is related to the expected risk, is
close to || · ||m, which corresponds to the empirical risk. Motivated by this property, we aim to find
a shallow ReLU network ϕ¯p with small expected risk by solving the empirical risk minimization
problem:
min
¯p∈ ¯P
∥ϕ¯p − y∥2
m.
Since the set Φ ¯P of shallow ReLU networks is non-convex, this minimization cannot be solved
with efficient convex optimizers. Therefore, instead of analyzing only the solution ϕ∗
¯p of the opti-
mization problem, we introduce a tolerance ϵ > 0 for the empirical risk and provide bounds on the
generalization error, which hold uniformly on the sublevel set
¯Qy,ϵ := ¯p ∈ ¯P : ∥ϕ¯p − y∥2
m ≤ ϵ .
Before considering generic regression problems, we will initially assume the label y to be a neural
network itself, parameterized by a tuple p∗ within the hypothesis set P . For all (x, y) in the support of
µ, we have y = ϕp∗ (x) and the expected risk’s minimum on P is zero. Using the sufficient condition
for NeuRIPs from Theorem 1, we can provide generalization bounds for ϕ¯p ∈ ¯Qy,ϵ for any ϵ > 0.
Theorem 2. Let ¯P be a parameter set that satisfies Assumption 1 and let u ≥ 2 and t ≥ ϵ > 0 be
constants. Furthermore, let the number m of samples satisfy
m ≥ 8n3c2
w (8cb + d + ln(2)) max

C1
u
(t − ϵ)2 , C2
n2c2
wu
(t − ϵ)2

,
where C1 and C2 are universal constants. Let {(xj , yj )}m
j=1 be a dataset respecting Assumption 2
and let there exist a ¯p∗ ∈ ¯P such that yj = ϕ¯p∗ (xj ) holds for all j ∈ [m]. Then, with probability at
least 1 − 17 exp(−u/4), we have for all ¯q ∈ ¯Qy,ϵ that
∥ϕ¯q − ϕ¯p∗ ∥2
µ ≤ t.
Proof. We notice that ¯Qy,ϵ is a set of shallow neural networks with 2n neurons. We normalize such
networks with a function norm greater than t and parameterize them by
¯Rt := {ϕ¯p − ϕ¯p∗ : ¯p ∈ ¯P , ∥ϕ¯p − ϕ¯p∗ ∥µ > t}.
We assume that NeuRIPs( ¯Rt) holds for s = (t − ϵ)2/t2. In this case, for all ¯q ∈ ¯Qy,ϵ, we have that
∥ϕ¯q − ϕ¯p∗ ∥m ≥ t and thus ¯q /∈ ¯Qϕ¯p∗ ,ϵ, which implies that ∥ϕ¯q − ϕ¯p∗ ∥µ ≤ t.
We also note that ¯Rt satisfies Assumption 1 with a rescaled constant cw/t and normalization-invariant
cb, if ¯P satisfies it for cw and cb. Theorem 1 gives a lower bound on the sample complexity for
NeuRIPs( ¯Rt), completing the proof.
At any network where an optimization method terminates, the concentration of the empirical risk
at the expected risk can be achieved with less data than needed to achieve an analogous NeuRIPs
event. However, in the chosen stochastic setting, we cannot assume that the termination of an
optimization and the norm concentration at that network are independent events. We overcome this
by not specifying the outcome of an optimization method and instead stating uniform bounds on
the norm concentration. The only assumption on an algorithm is therefore the identification of a
network that permits an upper bound ϵ on its empirical risk. The event NeuRIPs( ¯Rt) then restricts the
expected risk to be below the corresponding level t.
We now discuss the empirical risk surface for generic distributions µ that satisfy Assumption 2, where
y does not necessarily have to be a neural network.
Theorem 3. There exist constants C0, C1, C2, C3, C4, and C5 such that the following holds: Let ¯P
satisfy Assumption 1 for some constants cw, cb, and let ¯p∗ ∈ ¯P be such that for some c¯p∗ ≥ 0 we
have
Eµ

exp
 (y − ϕ¯p∗ (x))2
c2
¯p∗

≤ 2.
We assume, for any s ∈ (0, 1) and confidence parameter u > 0, that the number of samples m is
large enough such that
m ≥ 8
(1 − s)2 max

C1
 n3c2
w(8cb + d + ln(2))
u

, C2n2c2
w
 u
s

.
4
We further select confidence parameters v1, v2 > C0, and define for some ω ≥ 0 the parameter
η := 2(1 − s)∥ϕ¯p∗ − y∥µ + C3v1v2c¯p∗
1
(1 − s)1/4 + ω√1 − s.
If we set ϵ = ∥ϕ¯p∗ − y∥2
m + ω2 as the tolerance for the empirical risk, then the probability that all
¯q ∈ ¯Qy,ϵ satisfy
∥ϕ¯q − y∥µ ≤ η
is at least
1 − 17 exp

− u
4

− C5v2 exp

− C4mv2
2
2

.
Proof sketch. (Complete proof in Appendix E) We first define and decompose the excess risk by
E(¯q, ¯p∗) := ∥ϕ¯q − y∥2
µ − ∥ϕ¯p∗ − y∥2
µ = ∥ϕ¯q − ϕ¯p∗ ∥2
µ − 2
m
mX
j=1
(ϕ¯p∗ (xj ) − yj )(ϕ¯q (xj ) − ϕ¯p∗ (xj )).
It suffices to show, that within the stated confidence level we have ∥ϕ¯q − y∥µ > η . This implies the
claim since ∥ϕ¯q − y∥m ≤ ϵ implies ∥ϕ¯q − y∥µ ≤ η. We have E[E(¯q, ¯p∗)] > 0. It now only remains
to strengthen the condition on η > 3∥ϕ¯p∗ − y∥µ to achieve E(¯q, ¯p∗) > ω2. We apply Theorem 1
to derive a bound on the fluctuation of the first term. The concentration rate of the second term is
derived similar to Theorem 1 by using chaining techniques. Finally in Appendix E, Theorem 12 gives
a general bound to achieve
E(¯q, ¯p∗) > ω2
uniformly for all ¯q with ∥ϕ¯q − ϕ¯p∗ ∥µ > η. Theorem 3 then follows as a simplification.
It is important to notice that, in Theorem 3, as the data size m approaches infinity, one can select
an asymptotically small deviation constant s. In this limit, the bound η on the generalization error
converges to 3∥ϕ¯p∗ − y∥µ + ω. This reflects a lower limit of the generalization bound, which is the
sum of the theoretically achievable minimum of the expected risk and the additional tolerance ω.
The latter is an upper bound on the empirical risk, which real-world optimization algorithms can be
expected to achieve.
5 Size Control of Stochastic Processes on Shallow Networks
In this section, we introduce the key techniques for deriving concentration statements for the em-
pirical norm, uniformly valid for sets of shallow ReLU networks. We begin by rewriting the event
NeuRIPs( ¯P ) by treating µ as a stochastic process, indexed by the parameter set ¯P . The event
NeuRIPs( ¯P ) holds if and only if we have
sup
¯p∈ ¯P
|∥ϕ¯p∥m − ∥ϕ¯p∥µ| ≤ s sup
¯p∈ ¯P
∥ϕ¯p∥µ.
The supremum of stochastic processes has been studied in terms of their size. To determine the size
of a process, it is essential to determine the correlation between its variables. To this end, we define
the Sub-Gaussian metric for any parameter tuples ¯p, ¯q ∈ ¯P as
dψ2 (ϕ¯p, ϕ¯q ) := inf
(
Cψ2 ≥ 0 : E
"
exp |ϕ¯p(x) − ϕ¯q (x)|2
C2
ψ2
!#
≤ 2
)
.
A small Sub-Gaussian metric between random variables indicates that their values are likely to be
close. To capture the Sub-Gaussian structure of a process, we introduce ϵ-nets in the Sub-Gaussian
metric. For a given ϵ > 0, these are subsets ¯Q ⊆ ¯P such that for every ¯p ∈ ¯P , there is a ¯q ∈ ¯Q
satisfying
dψ2 (ϕ¯p, ϕ¯q ) ≤ ϵ.
The smallest cardinality of such an ϵ-net ¯Q is known as the Sub-Gaussian covering number
N (Φ ¯P , dψ2 , ϵ). The next Lemma offers a bound for such covering numbers specific to shallow
ReLU networks.
5
Lemma 1. Let ¯P be a parameter set satisfying Assumption 1. Then there exists a set ˆP with ¯P ⊆ ˆP
such that
N (Φ ˆP , dψ2 , ϵ) ≤ 2n ·
 16ncbcw
ϵ + 1
n
·
 32ncbcw
ϵ + 1
n
·
 1
ϵ sin
 1
16ncw

+ 1
d
.
The proof of this Lemma is based on the theory of stochastic processes and can be seen in Theorem 8
of Appendix C.
To obtain bounds of the form (6) on the size of a process, we use the generic chaining method. This
method offers bounds in terms of the Talagrand-functional of the process in the Sub-Gaussian metric.
We define it as follows. A sequence T = (Tk)k∈N0 in a set T is admissible if T0 = 1 and Tk ≤ 2(2k ).
The Talagrand-functional of the metric space is then defined as
γ2(T, d) := inf
(Tk ) sup
t∈T
∞X
k=0
2kd(t, Tk),
where the infimum is taken across all admissible sequences.
With the bounds on the Sub-Gaussian covering number from Lemma 1, we provide a bound on the
Talagrand-functional for shallow ReLU networks in the following Lemma. This bound is expected to
be of independent interest.
Lemma 2. Let ¯P satisfy Assumption 1. Then we have
γ2(Φ ¯P , dψ2 ) ≤
r 2
π
 8n3/2cw(8cb + d + 1)
ln(2)
p2 ln(2)

.
The key ideas to show this bound are similar to the ones used to prove Theorem 9 in Appendix C.
To provide bounds for the empirical process, we use the following Lemma, which we prove in
Appendix D.
Lemma 3. Let Φ be a set of real functions, indexed by a parameter set ¯P and define
N (Φ) :=
Z ∞
0
q
ln N (Φ, dψ2 , ϵ)dϵ and ∆(Φ) := sup
ϕ∈Φ
∥ϕ∥ψ2 .
Then, for any u ≥ 2, we have with probability at least 1 − 17 exp(−u/4) that
sup
ϕ∈Φ
|∥ϕ∥m − ∥ϕ∥µ| ≤ u
√m

N (Φ) + 10
3 ∆(Φ)

.
The bounds on the sample complexity for achieving the NeuRIPs event, from Theorem 1, are proven
by applying these Lemmata.
Proof of Theorem 1. Since we assume ||ϕ¯p||µ > 1 for all ¯p ∈ ¯P , we have
sup
¯p∈ ¯P
|∥ϕ¯p∥m − ∥ϕ¯p∥µ| ≤ sup
¯p∈ ¯P
|∥ϕ¯p∥m − ∥ϕ¯p∥µ|/∥ϕ¯p∥µ.
Applying Lemma 3, and further applying the bounds on the covering numbers and the Talagrand-
functional for shallow ReLU networks, the NeuRIPs( ¯P ) event holds in case of s > 3. The sample
complexities that are provided in Theorem 1 follow from a refinement of this condition.
6 Uniform Generalization of Sublevel Sets of the Empirical Risk
In case of the NeuRIPs event, the function norm || · ||µ corresponding to the expected risk is close
to || · ||m, which corresponds to the empirical risk. With the previous results, we can now derive
uniform generalization error bounds in the sublevel set of the empirical risk.
We use similar techniques and we define the following sets.
∥f ∥p = sup
1≤q≤p
∥f ∥q
Λk0,u = inf
(Tk ) sup
f ∈F
∞X
k0
2k∥f − Tk(f )∥u2k
6
and we need the following lemma:
Lemma 9. For any set F of functions and u ≥ 1, we have
Λ0,u(F ) ≤ 2√e(γ2(F, dψ2 ) + ∆(F )).
Theorem 10. Let P be a parameter set satisfying Assumption 1. Then, for any u ≥ 1, we have with
probability at least 1 − 17 exp(−u/4) that
sup
¯p∈P
∥ϕ¯p∥m − ∥ϕ¯p∥µ ≤ u
√m

16n3/2cw(8cb + d + 1) + 2ncw

.
Proof. To this end we have to bound the Talagrand functional, where we can use Dudley’s inequality
(Lemma 6). To finish the proof, we apply the bounds on the covering numbers provided by Theorem
6.
Theorem 11. Let ¯P ⊆ (Rd × R × ±1)n satisfy Assumption 1. Then there exist universal constants
C1, C2 such that
sup
¯p∈P
∥ϕ¯p∥m − ∥ϕ¯p∥µ ≤
r 2
π
 8n3/2cw(8cb + d + 1)
ln(2)
p2 ln(2)

.
7 Conclusion
In this study, we investigated the empirical risk surface of shallow ReLU networks in terms of uniform
concentration events for the empirical norm. We defined the Neural Restricted Isometry Property
(NeuRIPs) and determined the sample complexity required to achieve NeuRIPs, which depends on
realistic parameter bounds and the network architecture. We applied our findings to derive upper
bounds on the expected risk, which are valid uniformly across sublevel sets of the empirical risk.
If a network optimization algorithm can identify a network with a small empirical risk, our results
guarantee that this network will generalize well. By deriving uniform concentration statements, we
have resolved the problem of independence between the termination of an optimization algorithm at
a certain network and the empirical risk concentration at that network. Future studies may focus on
performing uniform empirical norm concentration on the critical points of the empirical risk, which
could lead to even tighter bounds for the sample complexity.
We also plan to apply our methods to input distributions more general than the Gaussian distribution.
If generic Gaussian distributions can be handled, one could then derive bounds for the Sub-Gaussian
covering number for deep ReLU networks by induction across layers. We also expect that our
results on the covering numbers could be extended to more generic Lipschitz continuous activation
functions other than ReLU. This proposition is based on the concentration of measure phenomenon,
which provides bounds on the Sub-Gaussian norm of functions on normal concentrating input spaces.
Because these bounds scale with the Lipschitz constant of the function, they can be used to find ϵ-nets
for neurons that have identical activation patterns.
Broader Impact
Supervised machine learning now affects both personal and public lives significantly. Generalization is
critical to the reliability and safety of empirically trained models. Our analysis aims to achieve a deeper
understanding of the relationships between generalization, architectural design, and available data.
We have discussed the concepts and demonstrated the effectiveness of using uniform concentration
events for generalization guarantees of common supervised machine learning algorithms.
"""
content_r015 = """
Examining the Convergence of Denoising Diffusion Probabilistic
Models: A Quantitative Analysis
Abstract
Deep generative models, particularly diffusion models, are a significant family within deep learning. This study
provides a precise upper limit for the Wasserstein distance between a learned distribution by a diffusion model
and the target distribution. In contrast to earlier research, this analysis does not rely on presumptions regarding
the learned score function. Furthermore, the findings are applicable to any data-generating distributions within
restricted instance spaces, even those lacking a density relative to the Lebesgue measure, and the upper limit is not
exponentially dependent on the ambient space dimension. The primary finding expands upon recent research by
Mbacke et al. (2023), and the proofs presented are fundamental.
1 Introduction
Diffusion models, alongside generative adversarial networks and variational autoencoders (VAEs), are among the most influential
families of deep generative models. These models have demonstrated remarkable empirical results in generating images and audio,
as well as in various other applications.
Two primary methods exist for diffusion models: denoising diffusion probabilistic models (DDPMs) and score-based generative
models (SGMs). DDPMs incrementally convert samples from the desired distribution into noise via a forward process, while
simultaneously training a backward process to reverse this transformation, enabling the creation of new samples. Conversely, SGMs
employ score-matching methods to approximate the score function of the data-generating distribution, subsequently generating new
samples through Langevin dynamics. Recognizing that real-world distributions might lack a defined score function, adding varying
noise levels to training samples to encompass the entire instance space and training a neural network to concurrently learn the score
function for all noise levels has been proposed.
Although DDPMs and SGMs may initially seem distinct, it has been demonstrated that DDPMs implicitly approximate the score
function, with the sampling process resembling Langevin dynamics. Moreover, a unified perspective of both methods using stochastic
differential equations (SDEs) has been derived. The SGM can be viewed as a discretization of Brownian motion, and the DDPM as a
discretization of an Ornstein-Uhlenbeck process. Consequently, both DDPMs and SGMs are commonly referred to as SGMs in the
literature. This explains why prior research investigating the theoretical aspects of diffusion models has adopted the score-based
framework, necessitating assumptions about the effectiveness of the learned score function.
In this research, a different strategy is employed, applying methods created for VAEs to DDPMs, which can be viewed as hierarchical
VAEs with fixed encoders. This method enables the derivation of quantitative, Wasserstein-based upper bounds without making
assumptions about the data distribution or the learned score function, and with simple proofs that do not need the SDE toolkit.
Furthermore, the bounds presented here do not involve any complex discretization steps, as the forward and backward processes are
considered discrete-time from the beginning, rather than being viewed as discretizations of continuous-time processes.
1.1 Related Works
There has been an increasing amount of research aimed at providing theoretical findings on the convergence of SGMs. However,
these studies frequently depend on restrictive assumptions regarding the data-generating distribution, produce non-quantitative upper
bounds, or exhibit exponential dependencies on certain parameters. This work successfully circumvents all three of these limitations.
Some bounds are based on very restrictive assumptions about the data-generating distribution, such as log-Sobolev inequalities,
which are unrealistic for real-world data distributions. Furthermore, some studies establish upper bounds on the Kullback-Leibler
(KL) divergence or the total variation (TV) distance between the data-generating distribution and the distribution learned by the
diffusion model; however, unless strong assumptions are made about the support of the data-generating distribution, KL and TV
reach their maximum values. Such assumptions arguably do not hold for real-world data-generating distributions, which are widely
believed to satisfy the manifold hypothesis. Other work establishes conditions under which the support of the input distribution
is equal to the support of the learned distribution, and generalizes the bound to all f-divergences. Assuming L2 accurate score
estimation, some establish Wasserstein distance upper bounds under weaker assumptions on the data-generating distribution, but
their Wasserstein-based bounds are not quantitative. Quantitative Wasserstein distance upper bounds under the manifold hypothesis
have been derived, but these bounds exhibit exponential dependencies on some of the problem parameters.
1.2 Our contributions
In this study, strong assumptions about the data-generating distribution are avoided, and a quantitative upper bound on the Wasserstein
distance is established without exponential dependencies on problem parameters, including the ambient space dimension. Moreover,
a common aspect of the aforementioned studies is that their bounds are contingent on the error of the score estimator. According to
some, providing precise guarantees for the estimation of the score function is challenging, as it necessitates an understanding of the
non-convex training dynamics of neural network optimization, which is currently beyond reach. Therefore, upper bounds are derived
without making assumptions about the learned score function. Instead, the bound presented here is dependent on a reconstruction
loss calculated over a finite independent and identically distributed (i.i.d.) sample. Intuitively, a loss function is defined, which
quantifies the average Euclidean distance between a sample from the data-generating distribution and the reconstruction obtained by
sampling noise and passing it through the backward process (parameterized by ˘03b8). This method is inspired by previous work on
VAEs.
This approach offers numerous benefits: it does not impose restrictive assumptions on the data-generating distribution, avoids
exponential dependencies on the dimension, and provides a quantitative upper bound based on the Wasserstein distance. Furthermore,
this method benefits from utilizing very straightforward and basic proofs.
2 Preliminaries
Throughout this paper, lowercase letters are used to represent both probability measures and their densities with respect to the
Lebesgue measure, and variables are added in parentheses to enhance readability (e.g., q(xt|xt−1) to denote a time-dependent
conditional distribution). An instance space X, which is a subset of RD with the Euclidean distance as the underlying metric, and
a target data-generating distribution µ ∈ M +
1 (X) are considered. Note that it is not assumed that µ has a density with respect to
the Lebesgue measure. Additionally, || · || represents the Euclidean (L2) norm, and Ep(x) is used as shorthand for Ex∼p(x). Given
probability measures p, q ∈ M +
1 (X) and a real number k > 1, the Wasserstein distance of order k is defined as (Villani, 2009):
Wk(p, q) = inf
γ∈Γ(p,q)
Z
X×X
||x − y||kdγ(x, y)
1/k
,
where Γ(p, q) denotes the set of couplings of p and q, meaning the set of joint distributions on X × X with respective marginals p
and q. The product measure p ⊗ q is referred to as the trivial coupling, and the Wasserstein distance of order 1 is simply referred to
as the Wasserstein distance.
2.1 Denoising Diffusion Models
Instead of employing the SDE framework, diffusion models are presented using the DDPM formulation with discrete-time processes.
A diffusion model consists of two discrete-time stochastic processes: a forward process and a backward process. Both processes are
indexed by time 0 ≤ t ≤ T , where the number of time steps T is a predetermined choice.
**The forward process.** The forward process transforms a data point x0 ∼ µ into a noise distribution q(xT |x0) through a sequence
of conditional distributions q(xt|xt−1) for 1 ≤ t ≤ T . It is assumed that the forward process is defined such that for sufficiently
large T , the distribution q(xT |x0) is close to a simple noise distribution p(xT ), which is referred to as the prior distribution. For
instance, p(xT ) = N (xT ; 0, I), the standard multivariate normal distribution, has been chosen in previous work.
**The backward process.** The backward process is a Markov process with parametric transition kernels. The objective of the
backward process is to perform the reverse operation of the forward process: transforming noise samples into (approximate) samples
from the distribution µ. Following previous work, it is assumed that the backward process is defined by Gaussian distributions
pθ (xt−1|xt) for 2 ≤ t ≤ T as
pθ (xt−1|xt) = N (xt−1; gθ
t (xt), σ2
t I),
and
pθ (x0|x1) = gθ
1 (x1),
where the variance parameters σ2
t ∈ R≥0 are defined by a fixed schedule, the mean functions gθ
t : RD → RD are learned using a
neural network (with parameters θ) for 2 ≤ t ≤ T , and gθ
1 : RD → X is a separate function dependent on σ1. In practice, the same
network has been used for the functions gθ
t for 2 ≤ t ≤ T , and a separate discrete decoder for gθ
1 .
2
Generating new samples from a trained diffusion model is accomplished by sampling xt−1 ∼ pθ (xt−1|xt) for 1 ≤ t ≤ T , starting
from a noise vector xT ∼ p(xT ) sampled from the prior p(xT ).
The following assumption is made regarding the backward process.
**Assumption 1.** It is assumed that for each 1 ≤ t ≤ T , there exists a constant Kθ
t > 0 such that for every x1, x2 ∈ X,
||gθ
t (x1) − gθ
t (x2)|| ≤ Kθ
t ||x1 − x2||.
In other words, gθ
t is Kθ
t -Lipschitz continuous. This assumption is discussed in Remark 3.2.
2.2 Additional Definitions
The distribution πθ (·|x0) is defined as
πθ (·|x0) = q(xT |x0)pθ (xT −1|xT )pθ (xT −2|xT −1) . . . pθ (x1|x2)pθ (·|x1).
Intuitively, for each x0 ∈ X, πθ (·|x0) represents the distribution on X obtained by reconstructing samples from q(xT |x0) through
the backward process. Another way to interpret this distribution is that for any function f : X → R, the following equation holds:
Eπθ (ˆx0|x0)[f (ˆx0)] = Eq(xT |x0)Epθ (xT −1|xT ) . . . Epθ (x1|x2)Epθ (ˆx0|x1)[f (ˆx0)].
Given a finite set S = {x1
0, . . . , xn
0 } i.i.d. ∼ µ, the regenerated distribution is defined as the following mixture:
µθ
n = 1
n
nX
i=1
πθ (·|xi
0).
This definition is analogous to the empirical regenerated distribution defined for VAEs. The distribution on X learned by the
diffusion model is denoted as πθ (·) and defined as
πθ (·) = p(xT )pθ (xT −1|xT )pθ (xT −2|xT −1) . . . pθ (x1|x2)pθ (·|x1).
In other words, for any function f : X → R, the expectation of f with respect to πθ (·) is
Eπθ (ˆx0)[f (ˆx0)] = Ep(xT )Epθ (xT −1|xT ) . . . Epθ (x1|x2)Epθ (ˆx0|x1)[f (ˆx0)].
Hence, both πθ (·) and πθ (·|x0) are defined using the backward process, with the difference that πθ (·) starts with the prior
p(xT ) = N (xT ; 0, I), while πθ (·|x0) starts with the noise distribution q(xT |x0).
Finally, the loss function lθ : X × X → R is defined as
lθ (xT , x0) = Epθ (xT −1|xT )Epθ (xT −2|xT −1) . . . Epθ (x1|x2)Epθ (ˆx0|x1)[||x0 − ˆx0||].
Hence, given a noise vector xT and a sample x0, the loss lθ (xT , x0) represents the average Euclidean distance between x0 and any
sample obtained by passing xT through the backward process.
2.3 Our Approach
The goal is to upper-bound the distance W1(µ, πθ (·)). Since the triangle inequality implies
W1(µ, πθ (·)) ≤ W1(µ, µθ
n) + W1(µθ
n, πθ (·)),
the distance W1(µ, πθ (·)) can be upper-bounded by upper-bounding the two expressions on the right-hand side separately. The
upper bound on W1(µ, µθ
n) is obtained using a straightforward adaptation of a proof. First, W1(µ, µθ
n) is upper-bounded using the
expectation of the loss function lθ , then the resulting expression is upper-bounded using a PAC-Bayesian-style expression dependent
on the empirical risk and the prior-matching term.
The upper bound on the second term W1(µθ
n, πθ (·)) uses the definition of µθ
n. Intuitively, the difference between πθ (·|xi
0) and πθ (·)
is determined by the corresponding initial distributions: q(xT |xi
0) and p(xT ) for πθ (·). Hence, if the two initial distributions are
close, and if the steps of the backward process are smooth (see Assumption 1), then πθ (·|xi
0) and πθ (·) are close to each other.
3
3 Main Result
3.1 Theorem Statement
We are now ready to present the main result: a quantitative upper bound on the Wasserstein distance between the data-generating
distribution µ and the learned distribution πθ (·).
**Theorem 3.1.** Assume the instance space X has finite diameter ∆ = supx,x′∈X ||x − x′|| < ∞, and let λ > 0 and δ ∈ (0, 1) be
real numbers. Using the definitions and assumptions of the previous section, the following inequality holds with probability at least
1 − δ over the random draw of S = {x1
0, . . . , xn
0 } i.i.d. ∼ µ:
W1(µ, πθ (·)) ≤ 1
n
nX
i=1
Eq(xT |xi
0)[lθ (xT , xi
0)] + 1
λn
nX
i=1
KL(q(xT |xi
0)||p(xT )) + 1
λn log n
δ + λ∆2
8n
+
TY
t=1
Kθ
t
!
Eq(xT |xi
0)Ep(yT )[||xT − yT ||]
+
TX
t=2
t−1Y
i=1
Kθ
i
!
σtEϵ,ϵ′ [||ϵ − ϵ′||],
where ϵ, ϵ′ ∼ N (0, I) are standard Gaussian vectors.
**Remark 3.1.** Before presenting the proof, let us discuss Theorem 3.1.
* Because the right-hand side of the equation depends on a quantity computed using a finite i.i.d. sample S, the bound holds with
high probability with respect to the randomness of S. This is the price we pay for having a quantitative upper bound with no
exponential dependencies on problem parameters and no assumptions on the data-generating distribution µ. * The first term of the
right-hand side is the average reconstruction loss computed over the sample S = {x1
0, . . . , xn
0 }. Note that for each 1 ≤ i ≤ n, the
expectation of lθ (xT |xi
0) is only computed with respect to the noise distribution q(xT |xi
0) defined by xi
0 itself. Hence, this term
measures how well a noise vector xT ∼ q(xT |xi
0) recovers the original sample xi
0 using the backward process, and averages over
the set S = {x1
0, . . . , xn
0 }. * If the Lipschitz constants satisfy Kθ
t < 1 for all 1 ≤ t ≤ T , then the larger T is, the smaller the upper
bound gets. This is because the product of Kθ
t ’s then converges to 0. In Remark 3.2 below, we show that the assumption that Kθ
t < 1
for all t is a quite reasonable one. * The hyperparameter λ controls the trade-off between the prior-matching (KL) term and the
diameter term ∆2. If Kθ
t < 1 for all 1 ≤ t ≤ T and T → ∞, then the convergence of the bound largely depends on the choice of λ.
In that case, λ ∝ n1/2 leads to faster convergence, while λ ∝ n leads to slower convergence to a smaller quantity. This is because
the bound stems from PAC-Bayesian theory, where this trade-off is common. * The last term of the equation does not depend on the
sample size n. Hence, the upper bound given by Theorem 3.1 does not converge to 0 as n → ∞. However, if the Lipschitz factors
(Kθ
t )1≤t≤T are all less than 1, then this term can be very small, especially in low-dimensional spaces.
3.2 Proof of the main theorem
The following result is an adaptation of a previous result.
**Lemma 3.2.** Let λ > 0 and δ ∈ (0, 1) be real numbers. With probability at least 1 − δ over the randomness of the sample
S = {x1
0, . . . , xn
0 } i.i.d. ∼ µ, the following holds:
W1(µ, µθ
n) ≤ 1
n
nX
i=1
Eq(xT |xi
0)[lθ (xT , xi
0)] + 1
λn
nX
i=1
KL(q(xT |xi
0)||p(xT )) + 1
λn log n
δ + λ∆2
8n .
The proof of this result is a straightforward adaptation of a previous proof.
Now, let us focus our attention on the second term of the right-hand side of the equation, namely W1(µθ
n, πθ (·)). This part is trickier
than for VAEs, for which the generative model’s distribution is simply a pushforward measure. Here, we have a non-deterministic
sampling process with T steps.
Assumption 1 leads to the following lemma on the backward process.
**Lemma 3.3.** For any given x1, y1 ∈ X, we have
Epθ (x0|x1)Epθ (y0|y1)[||x0 − y0||] ≤ Kθ
1 ||x1 − y1||.
Moreover, if 2 ≤ t ≤ T , then for any given xt, yt ∈ X, we have
4
Epθ (xt−1|xt)Epθ (yt−1|yt)[||xt−1 − yt−1||] ≤ Kθ
t ||xt − yt|| + σtEϵ,ϵ′ [||ϵ − ϵ′||],
where ϵ, ϵ′ ∼ N (0, I), meaning Eϵ,ϵ′ is a shorthand for Eϵ,ϵ′∼N (0,I).
**Proof.** For the first part, let x1, y1 ∈ X. Since according to the equation pθ (x0|x1) = δgθ
1 (x1)(x0) and pθ (y0|y1) = δgθ
1 (y1)(y0),
then
Epθ (x0|x1)Epθ (y0|y1)[||x0 − y0||] = ||gθ
1 (x1) − gθ
1 (y1)|| ≤ Kθ
1 ||x1 − y1||.
For the second part, let 2 ≤ t ≤ T and xt, yt ∈ X. Since pθ (xt−1|xt) = N (xt−1; gθ
t (xt), σ2
t I), the reparameterization trick implies
that sampling xt−1 ∼ pθ (xt−1|xt) is equivalent to setting
xt−1 = gθ
t (xt) + σtϵt, with ϵt ∼ N (0, I).
Using the above equation, the triangle inequality, and Assumption 1, we obtain
Epθ (xt−1|xt)Epθ (yt−1|yt)[||xt−1 − yt−1||]
= Eϵt,ϵ′
t∼N (0,I)[||gθ
t (xt) + σtϵt − gθ
t (yt) − σtϵ′
t||]
≤ Eϵt,ϵ′
t∼N (0,I)[||gθ
t (xt) − gθ
t (yt)||] + σtEϵt,ϵ′
t∼N (0,I)[||ϵt − ϵ′
t||]
≤ Kθ
t ||xt − yt|| + σtEϵ,ϵ′ [||ϵ − ϵ′||],
where ϵ, ϵ′ ∼ N (0, I).
Next, we can use the inequalities of Lemma 3.3 to prove the following result.
**Lemma 3.4.** Let T ≥ 1. The following inequality holds:
Epθ (xT −1|xT )Epθ (yT −1|yT )Epθ (xT −2|xT −1)Epθ (yT −2|yT −1) . . . Epθ (x0|x1)Epθ (y0|y1)[||x0 − y0||]
≤
TY
t=1
Kθ
t
!
||xT − yT || +
TX
t=2
t−1Y
i=1
Kθ
i
!
σtEϵ,ϵ′ [||ϵ − ϵ′||],
where ϵ, ϵ′ ∼ N (0, I).
**Proof Idea.** Lemma 3.4 is proven by induction using Lemma 3.3 in the induction step.
Using the two previous lemmas, we obtain the following upper bound on W1(µθ
n, πθ (·)).
**Lemma 3.5.** The following inequality holds:
W1(µθ
n, πθ (·)) ≤ 1
n
nX
i=1
TY
t=1
Kθ
t
!
Eq(xT |xi
0)Ep(yT )[||xT − yT ||] +
TX
t=2
t−1Y
i=1
Kθ
i
!
σtEϵ,ϵ′ [||ϵ − ϵ′||],
where ϵ, ϵ′ ∼ N (0, I).
**Proof.** Using the definition of W1, the trivial coupling, the definitions of µθ
n and πθ (·), and Lemma 3.4, we get the desired result.
Combining Lemmas 3.2 and 3.5 with the triangle inequality yields Theorem 3.1.
3.3 Special case using the forward process of Ho et al. (2020)
Theorem 3.1 establishes a general upper bound that holds for any forward process, as long as the backward process satisfies
Assumption 1. In this section, we specialize the statement of the theorem to the particular case of the forward process defined in
previous work.
Let X ⊆ RD . The forward process is a Gauss-Markov process with transition densities defined as
q(xt|xt−1) = N (xt; √αtxt−1, (1 − αt)I),
where α1, . . . , αT is a fixed noise schedule such that 0 < αt < 1 for all t. This definition implies that at each time step 1 ≤ t ≤ T ,
5
q(xt|x0) = N (xt; √¯αtx0, (1 − ¯αt)I), with ¯αt =
tY
i=1
αi.
The optimization objective to train the backward process ensures that for each time step t, the distribution pθ (xt−1|xt) remains close
to the ground-truth distribution q(xt−1|xt, x0) given by
q(xt−1|xt, x0) = N (xt−1; ˜µq
t (xt, x0), ˜σ2
t I),
where
˜µq
t (xt, x0) =
√αt(1 − ¯αt−1)
1 − ¯αt
xt +
√¯αt−1(1 − αt)
1 − ¯αt
x0.
Now, we discuss Assumption 1 under these definitions.
**Remark 3.2.** We can get a glimpse at the range of Kθ
t for a trained DDPM by looking at the distribution q(xt−1|xt, x0), since
pθ (xt−1|xt) is optimized to be as close as possible to q(xt−1|xt, x0).
For a given x0 ∼ µ, let us take a look at the Lipschitz norm of x 7 → ˜µq
t (x, x0). Using the above equation, we have
˜µq
t (xt, x0) − ˜µq
t (yt, x0) =
√αt(1 − ¯αt−1)
1 − ¯αt
(xt − yt).
Hence, x 7 → ˜µq
t (x, x0) is K′
t-Lipschitz continuous with
K′
t =
√αt(1 − ¯αt−1)
1 − ¯αt
.
Now, if αt < 1 for all 1 ≤ t ≤ T , then we have 1 − ¯αt > 1 − ¯αt−1, which implies K′
t < 1 for all 1 ≤ t ≤ T .
Remark 3.2 shows that the Lipschitz norm of the mean function ˜µq
t (·, x0) does not depend on x0. Indeed, looking at the previous
equation, we can see that for any initial x0, the Lipschitz norm K′
t = √αt(1− ¯αt−1)
1− ¯αt only depends on the noise schedule, not x0 itself.
Since gθ
t (·, x0) is optimized to match ˜µq
t (·, x0) for each x0 in the training set, and all the functions ˜µq
t (·, x0) have the same Lipschitz
norm K′
t, we believe it is reasonable to assume gθ
t is Lipschitz continuous as well. This is the intuition behind Assumption 1.
**The prior-matching term.** With the definitions of this section, the prior matching term KL(q(xT |x0)||p(xT )) has the following
closed form:
KL(q(xT |x0)||p(xT )) = 1
2
−D log(1 − ¯αT ) − D ¯αT + ¯αT ||x0||2 .
**Upper-bounds on the average distance between Gaussian vectors.** If ϵ, ϵ′ are D-dimensional vectors sampled from N (0, I), then
Eϵ,ϵ′ [||ϵ − ϵ′||] ≤ √2D.
Moreover, since q(xT |x0) = N (xT ; √¯αT x0, (1 − ¯αT )I) and the prior p(yT ) = N (yT ; 0, I),
Eq(xT |x0)Ep(yT )[||xT − yT ||] ≤ p¯αT ||x0||2 + (2 − ¯αT )D.
**Special case of the main theorem.** With the definitions of this section, the inequality of Theorem 3.1 implies that with probability
at least 1 − δ over the randomness of {x1
0, . . . , x
"""

prompt = f"""
I have a paper for you to analyze and categorize into one of the 5 conference categories you know.
Think step by step, building up on your reasonings and ending up concluding with the mmost appropriate conference for the paper.
Speak out your thinking and reasonings as you go through and think about various parts of the paper. Ensure thoughtts are rich, diverse, non-repetitive and after multiple thoughts(ensure ore than 5 thoughts upto 20 thoughts), you conclude withtthe final result as a one word answer which is the conference name.
You may crtique your previous thoughts, anticipate some futher reasonings to explore in later thoughts, contrast with other thoughts, and hence overall, have a cohesive and good thought process to come to a final well-reasoned conclusion.

Here is the paper's content:
{content_r015}
"""
print("Invoking llm.....")
response = model.invoke(
    [
        SystemMessage(content = sys_prommpt),
        HumanMessage(content = prompt)
    ]
)

print("Response from llm:")
print(response.content)