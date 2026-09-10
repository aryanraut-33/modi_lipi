# Preprocessing Methodology for Archaic Modi Script Translation

## Overview
Preprocessing acts as a foundational, critical mechanism within the broader architecture of translating historical documents. Given that archival manuscripts natively suffer from acute degradation, fading, and physical damage spanning centuries, raw imaging data contains vast quantities of visual noise. The primary role of the preprocessing pipeline is to aggressively isolate the target handwritten manuscript ink from complex backgrounds. By standardizing the continuous visual inputs into high-contrast morphological structures, the pipeline fundamentally reduces the latent variance that downstream Vision-Language Models must account for, directly improving their capacity to learn accurate transliteration mappings.

## Dataset Description (Pre-Preprocessing Context)
The raw dataset consists of continuous-tone digital captures of historical manuscripts written in the cursive, continuous Modi script. These documents, sourced primarily from governmental and historical archives, are characterized by severe physical and temporal degradation. Native characteristics include damp patches, fading ink, paper browning (foxing), ink bleed-through from the reverse sides of pages, script fragmentation, and extensive non-uniform illumination due to varied digitization conditions. The data is initially provided in multi-band Red-Green-Blue formats.

## Step-by-Step Preprocessing Techniques

### Grayscale Conversion
The initial stage of the pipeline transforms the multiband color image into a single-channel intensity image. By computing the weighted sum of color channels according to human perceptual luminances, we establish a standardized intensity map. This technique is computationally vital as the chromatic information in monochrome manuscript inks is largely redundant for structural decipherment. It dramatically accelerates subsequent processing stages by reducing dimensionality. The assumption is that the distinction between ink and paper background relies purely on relative intensity rather than hue.

Its mathematical formulation is defined as:
$$ Y = 0.299R + 0.587G + 0.114B $$

### Edge-Preserving Denoising (Bilateral Filtering)
To mitigate native paper textures and sensor noise without blurring the sharp stroke boundaries of the handwritten script, a non-linear bilateral filter is applied. Unlike standard Gaussian blurs, the bilateral filter utilizes both spatial proximity and intensity proximity to calculate average values. This technique is justified because historical documents contain high-frequency noise distributed across low-frequency background patches, coupled with high-frequency ink boundaries that absolutely must be preserved for accurate morphological extraction. It assumes that ink boundaries present a sharper intensity gradient than surrounding noise.

Its mathematical formulation at a given spatial location is:
$$ I^{filtered}(x) = \frac{1}{W_p} \sum_{x_i \in \Omega} I(x_i) \cdot f_r(\|I(x_i) - I(x)\|) \cdot g_s(\|x_i - x\|) $$
Where $W_p$ is the normalization factor, $f_r$ is the intensity range kernel, and $g_s$ is the spatial domain kernel.

### Adaptive Gaussian Thresholding
Global thresholding algorithms routinely fail on archival documents due to illumination gradients spanning the page. Adaptive Gaussian Thresholding resolves this by binarizing the image using a localized, dynamic threshold calculated over a highly specific pixel radius. This is critically justified as it normalizes dynamic lighting conditions, extracting faint text located in dark corners just as efficiently as bold text under bright illumination. It assumes that within any localized spatial block, there exists sufficient variance to calculate an accurate binarization threshold separating paper from ink.

Its mathematical formulation computes the local threshold and the corresponding mask:
$$ T(x, y) = \text{GaussianWeight}(x, y) * I_{window} - C $$
$$ I^{binary}(x, y) = \begin{cases} 255, & \text{if } I(x, y) > T(x, y) \\ 0, & \text{otherwise} \end{cases} $$

### Speckle and Artifact Removal via Connected Components
Following binarization, the image often features minute, disconnected dots caused by aged-paper artifacts or local binarization failures over stains. A Connected Components Analysis assesses the structural mass of contiguous pixels. Components failing to satisfy bounded minimum dimensional limits are computationally erased from the foreground mask. Given the cursive, continuous flow inherent to the Modi script, legitimate strokes present significant pixel counts and structural dimensions. Removing isolated pixel clusters forces the network to focus strictly on genuine graphemes rather than paper grain. This mathematically assumes that script ink guarantees an identifiable spatial mass exceeding basic thresholds.

Its mathematical formulation defines a preserved component belonging to the set of all objects $\mathcal{C}$ subject to dimensional evaluations:
$$ \text{Keep}(c) = \text{Area}(c) \geq \tau_{area} \lor (\text{Height}(c) \geq \tau_{h} \land \text{Width}(c) \geq \tau_{w}) $$

### False-Positive Foreground Suppression
Adaptive binarization algorithms frequently over-compensate, occasionally forcing heavy background stains into solid foreground objects. To suppress these false positives, the remaining component masks are juxtaposed against the original grayscale matrix to evaluate their underlying mean intensity. Background noise mapped as text frequently possesses a much lighter true intensity than actual carbon-based ink. Extinguishing components that fail this severity check ensures that margin stains are not passed into the vision encoder. The assumption holds that true ink strokes universally represent the darkest absolute values across the historic document.

Its mathematical formulation validates an extracted mask component cluster $M_c$:
$$ \frac{1}{|M_c|} \sum_{(x,y) \in M_c} I^{gray}(x,y) \leq \tau_{ink} $$

### Deskewing
To optimize the alignment of textual baselines for the neural network, the skew angle of the general foreground text can be calculated via minimum area bounding rectangles, followed by an affine rotation sequence. While theoretically pertinent for reducing spatial variance, deskewing is intentionally bypassed in the current highly-fragmented archival pipeline to circumvent catastrophic over-rotations fundamentally triggered by noise matrices sitting on the extremities of fragmented documents.

## Algorithmic and Implementation Details
The preprocessing pipeline is exclusively engineered inside Python 3, capitalizing on the OpenCV library for extremely efficient low-level C++ matrix operations and NumPy for rapid vectorized array evaluations. The bilateral filter implements a kernel diameter of 5 with substantial color and spatial sigma parameters rated at 100, effectively demanding aggressive smoothing while harshly terminating at stroke boundaries. The adaptive thresholding leverages a massive block dimension of 61 accompanied by a subtractive constant of 21, specifically scaled to accommodate high-resolution scans while filtering illumination sweeps. Geometric component filters impose firm base constraints demanding an area minimum of 100 pixels alongside rigid height and width metrics. Ultimately, the ink intensity suppressor imposes a strict cap of 130 on the 8-bit scale, eradicating lighter false-positives globally.

## Impact on Model Performance
Applying this preprocessing configuration substantially optimizes downstream modeling efforts. By explicitly terminating severe historical noise arrays and illuminating only structural strokes, the Vision-Language Model is completely unburdened from learning archaic data-drift and complex illumination invariances natively. Empirically, restricting cross-attention matrices strictly to graphemic shapes profoundly accelerates model convergence rates and ensures the calculated Character Error Rate directly maps to structural decipherment capacities as opposed to background misinterpretations.

## Reproducibility Considerations
To replicate the outcomes defined within this methodology, investigators must ensure identical initialization of algorithmic hyperparameters defined above. Raw image input scans must theoretically maintain parity with standard high-resolution archival parameters (e.g. 300 Dots Per Inch) to preserve the physical effect sizes underpinning the explicit geometric Connected Component thresholds. The pipeline inherently executes through a formalized operational sequence governed by deterministic parameters, eliminating probabilistic variance scaling during the translation mapping stages.

## References and Citations
[1] Kausadikar, S., et al. (2025). "Historic Scripts to Modern Vision," Journal of Historical Document Processing, vol. 14, no. 2, pp. 112-125.
[2] Tomasi, C., and Manduchi, R. (1998). "Bilateral Filtering for Gray and Color Images," Proceedings of the 1998 IEEE International Conference on Computer Vision, pp. 839-846.
[3] Bradley, D., and Roth, G. (2007). "Adaptive Thresholding using the Integral Image," Journal of Graphics Tools, vol. 12, no. 2, pp. 13-21.
