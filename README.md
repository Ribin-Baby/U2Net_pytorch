
# U2 NET

<div  align="center">
<img  src="docs/U2NETPR.png"  style="width: 85%">
<br>
<figcaption><i>Fig.1 - U2Net Architecture</i></figcaption>
</div>
<br>

* **U2-Net** is a two-level nested U-structure architecture. It uses a novel ReSidual U-block (RSU) module to extract multi-scale features without degrading resolution, allowing the network to go deeper and attain high resolution without significantly increasing memory and computation cost.
* used for for salient object detection, image segmentation, Image Matting, background removal and other image2image modeling tasks.
<div  align="center">
<img  src="docs/u2net.rsu-block.svg"  style="width: 80%">
<br>
<figcaption><i>Fig.2 - UNet or RSU Block</i></figcaption>
</div>
<br>

* U-Net is a U-shaped encoder-decoder architecture with residual connections between each layers. It captures contextual information and intricate detail.
* These U-Net blocks in U2Net architecture are called ReSidual U-block or RSU.

*  Example: we have trained an Image Matting model on [P3M-10k](https://paperswithcode.com/dataset/p3m-10k) dataset, and the results are given below.
<div  align="center">
<img  src="docs/example_u2net_Segment.png"  style="width: 60%">
<br>
<figcaption><i>Fig.3 - Image Matting  with U2-Net example</i></figcaption>
</div>
<br>

![training progress](docs/training_progress.gif)
<figcaption><i>Fig.4 - Image Matting with U2-Net training progress after each steps</i></figcaption>
