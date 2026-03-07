# Latent Inter-Frame Pruning: Exploiting Temporal Redundancy for Variable-Length Conditioned Video Generation 

Supplementary Material

[TOC]

## Comparisons to Existing Real-Time (Low Latency) V2V Models

Our method significantly increases the throughput of the base model (Self-Forcing <a href="#ref-1">[1]</a>) for real-time video editing while maintaining the visual quality and temporal consistency of videos.

- ***A beautiful blonde woman with blue eyes wearing is performing the moonwalk. Simple dark background.***

  <table>
    <tr>
      <th align="center">Input Video</th>
      <th align="center">LIF (Ours) - 21.3% Pruned</th>
      <th align="center">Self-Forcing <a href="#ref-1">[1]</a>)</th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/input/woman_walk.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/lif/woman_walk.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/self_forcing/woman_walk.mp4" /></td>
    </tr>
    <tr>
      <th align="center">StreamV2V <a href="#ref-2">[2]</a></th>
      <th align="center">StreamDiffusion <a href="#ref-3">[3]</a></th>
      <th align="center">ControlVideo <a href="#ref-4">[4]</a></th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/streamv2v/woman_walk.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/streamdiffusion/woman_walk.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/control_video/woman_walk.mp4" /></td>
    </tr>
  </table>

- ***A majestic lion is turning its head to look around in a field. Realistic fur texture, soft sunlight filtering through a white forest background.***

  <table>
    <tr>
      <th align="center">Input Video</th>
      <th align="center">LIF (Ours) - 11.9% Pruned</th>
      <th align="center">Self-Forcing <a href="#ref-1">[1]</a></th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/input/lion.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/lif/lion.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/self_forcing/lion.mp4" /></td>
    </tr>
    <tr>
      <th align="center">StreamV2V <a href="#ref-2">[2]</a></th>
      <th align="center">StreamDiffusion <a href="#ref-3">[3]</a></th>
      <th align="center">ControlVideo <a href="#ref-4">[4]</a></th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/streamv2v/lion.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/streamdiffusion/lion.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/control_video/lion.mp4" /></td>
    </tr>
  </table>

- ***A old man with white beard is holding and interacting with mysterious rock that has a small tree growing on it. Natural lighting, domestic interior background.***

  <table>
    <tr>
      <th align="center">Input Video</th>
      <th align="center">LIF (Ours) - 19.1% Pruned</th>
      <th align="center">Self-Forcing <a href="#ref-1">[1]</a></th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/input/old_man.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/lif/old_man.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/self_forcing/old_man.mp4" /></td>
    </tr>
    <tr>
      <th align="center">StreamV2V <a href="#ref-2">[2]</a></th>
      <th align="center">StreamDiffusion <a href="#ref-3">[3]</a></th>
      <th align="center">ControlVideo <a href="#ref-4">[4]</a></th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/streamv2v/old_man.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/streamdiffusion/old_man.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/control_video/old_man.mp4" /></td>
    </tr>
  </table>

- ***A vibrant red racer drifting on a simple grey asphalt track. Thick black outlines, bold flat colors, speed lines emphasizing the drift.***

  <table>
    <tr>
      <th align="center">Input Video</th>
      <th align="center">LIF (Ours) - 54.3% Pruned</th>
      <th align="center">Self-Forcing <a href="#ref-1">[1]</a></th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/input/race_car.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/lif/race_car.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/self_forcing/race_car.mp4" /></td>
    </tr>
    <tr>
      <th align="center">StreamV2V <a href="#ref-2">[2]</a></th>
      <th align="center">StreamDiffusion <a href="#ref-3">[3]</a></th>
      <th align="center">ControlVideo <a href="#ref-4">[4]</a></th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/streamv2v/race_car.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/streamdiffusion/race_car.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/control_video/race_car.mp4" /></td>
    </tr>
  </table>

- ***A woman wearing a black leather jacket riding a motorcycle while stretching her arms out joyfully. Realistic cinematic style, wind blowing through hair, blurred asphalt road beneath to imply speed.***

  <table>
    <tr>
      <th align="center">Input Video</th>
      <th align="center">LIF (Ours) - 20.7% Pruned</th>
      <th align="center">Self-Forcing <a href="#ref-1">[1]</a></th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/input/woman_biker.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/lif/woman_biker.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/self_forcing/woman_biker.mp4" /></td>
    </tr>
    <tr>
      <th align="center">StreamV2V <a href="#ref-2">[2]</a></th>
      <th align="center">StreamDiffusion <a href="#ref-3">[3]</a></th>
      <th align="center">ControlVideo <a href="#ref-4">[4]</a></th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/streamv2v/woman_biker.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/streamdiffusion/woman_biker.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/control_video/woman_biker.mp4" /></td>
    </tr>
  </table>

- ***A young anime protagonist riding a sleek bike with arms outstretched. 2D animation style, speed lines, vibrant colors, dramatic clouds in a blue sky background.***

  <table>
    <tr>
      <th align="center">Input Video</th>
      <th align="center">LIF (Ours) - 20.7% Pruned</th>
      <th align="center">Self-Forcing <a href="#ref-1">[1]</a></th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/input/biker.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/lif/biker.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/self_forcing/biker.mp4" /></td>
    </tr>
    <tr>
      <th align="center">StreamV2V <a href="#ref-2">[2]</a></th>
      <th align="center">StreamDiffusion <a href="#ref-3">[3]</a></th>
      <th align="center">ControlVideo <a href="#ref-4">[4]</a></th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/streamv2v/biker.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/streamdiffusion/biker.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/control_video/biker.mp4" /></td>
    </tr>
  </table>G

- ***Anime style animation of a frog dancing and performing acrobatic side somersaults. Vibrant cel-shaded colors.***

  <table>
    <tr>
      <th align="center">Input Video</th>
      <th align="center">LIF (Ours) - 16.8% Pruned</th>
      <th align="center">Self-Forcing <a href="#ref-1">[1]</a></th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/input/frog.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/lif/frog.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/self_forcing/frog.mp4" /></td>
    </tr>
    <tr>
      <th align="center">StreamV2V <a href="#ref-2">[2]</a></th>
      <th align="center">StreamDiffusion <a href="#ref-3">[3]</a></th>
      <th align="center">ControlVideo <a href="#ref-4">[4]</a></th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/streamv2v/frog.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/streamdiffusion/frog.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/control_video/frog.mp4" /></td>
    </tr>
  </table>
  
- ***Three corgi puppies sharing a meal together on a kitchen floor. Shallow depth of field with a soft, creamy bokeh background. The lighting is bright and airy.***

  <table>
    <tr>
      <th align="center">Input Video</th>
      <th align="center">LIF (Ours) - 33.8% Pruned</th>
      <th align="center">Self-Forcing <a href="#ref-1">[1]</a></th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/input/corgi.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/lif/corgi.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/self_forcing/corgi.mp4" /></td>
    </tr>
    <tr>
      <th align="center">StreamV2V <a href="#ref-2">[2]</a></th>
      <th align="center">StreamDiffusion <a href="#ref-3">[3]</a></th>
      <th align="center">ControlVideo <a href="#ref-4">[4]</a></th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/streamv2v/corgi.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/streamdiffusion/corgi.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/control_video/corgi.mp4" /></td>
    </tr>
  </table>

- ***Three cute, tigers eating together. Flat cel-shading, vibrant pastel colors.***

  <table>
    <tr>
      <th align="center">Input Video</th>
      <th align="center">LIF (Ours) - 33.8% Pruned</th>
      <th align="center">Self-Forcing <a href="#ref-1">[1]</a></th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/input/tiger.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/lif/tiger.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/self_forcing/tiger.mp4" /></td>
    </tr>
    <tr>
      <th align="center">StreamV2V <a href="#ref-2">[2]</a></th>
      <th align="center">StreamDiffusion <a href="#ref-3">[3]</a></th>
      <th align="center">ControlVideo <a href="#ref-4">[4]</a></th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/streamv2v/tiger.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/streamdiffusion/tiger.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/control_video/tiger.mp4" /></td>
    </tr>
  </table>

- ***Two cute, fluffy penguins wearing winter scarves waddling across a frozen ice path in Antarctica. Soft morning sunlight, glistening snow, National Geographic nature documentary style, realistic texture.***

  <table>
    <tr>
      <th align="center">Input Video</th>
      <th align="center">LIF (Ours) - 52.9% Pruned</th>
      <th align="center">Self-Forcing <a href="#ref-1">[1]</a></th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/input/penguin.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/lif/penguin.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/self_forcing/penguin.mp4" /></td>
    </tr>
    <tr>
      <th align="center">StreamV2V <a href="#ref-2">[2]</a></th>
      <th align="center">StreamDiffusion <a href="#ref-3">[3]</a></th>
      <th align="center">ControlVideo <a href="#ref-4">[4]</a></th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/streamv2v/penguin.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/streamdiffusion/penguin.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/control_video/penguin.mp4" /></td>
    </tr>
  </table>

## Comparisons with Training-free Pruning Methods

- ***Three cute, tigers eating together. Flat cel-shading, vibrant pastel colors.***

  <table>
    <tr>
      <th align="center">Input Video</th>
      <th align="center">LIF (Ours) - 33.8% Pruned</th>
      <th align="center">Self-Forcing <a href="#ref-1">[1]</a> - No Pruning</th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/input/tiger.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/lif/tiger.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/self_forcing/tiger.mp4" /></td>
    </tr>
    <tr>
      <th align="center">ToMe <a href="#ref-5">[5]</a> - 32% Pruned</th>
      <th align="center">Importance-based <a href="#ref-6">[6]</a> - 32% Pruned</th>
      <th align="center">IDF <a href="#ref-7">[7]</a> - 32% Pruned</th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/other_training_free_pruning/tome/tiger.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/other_training_free_pruning/importance_based/tiger.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/other_training_free_pruning/idf/tiger.mp4" /></td>
    </tr>
  </table>

- ***A vibrant red racer drifting on a simple grey asphalt track. Thick black outlines, bold flat colors, speed lines emphasizing the drift.***

  <table>
    <tr>
      <th align="center">Input Video</th>
      <th align="center">LIF (Ours) - 45.3% Pruned</th>
      <th align="center">Self-Forcing <a href="#ref-1">[1]</a> - No Pruning</th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/input/race_car.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/lif/race_car.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/self_forcing/race_car.mp4" /></td>
    </tr>
    <tr>
      <th align="center">ToMe <a href="#ref-5">[5]</a> - 32% Pruned</th>
      <th align="center">Importance-based <a href="#ref-6">[6]</a> - 32% Pruned</th>
      <th align="center">IDF <a href="#ref-7">[7]</a> - 32% Pruned</th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/other_training_free_pruning/tome/race_car.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/other_training_free_pruning/importance_based/race_car.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/other_training_free_pruning/idf/race_car.mp4" /></td>
    </tr>
  </table>

- ***Two cute, fluffy penguins wearing winter scarves waddling across a frozen ice path in Antarctica. Soft morning sunlight, glistening snow, National Geographic nature documentary style, realistic texture.***

  <table>
    <tr>
      <th align="center">Input Video</th>
      <th align="center">LIF (Ours) - 52.9% Pruned</th>
      <th align="center">Self-Forcing <a href="#ref-1">[1]</a> - No Pruning</th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/input/penguin.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/lif/penguin.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/diff_models/self_forcing/penguin.mp4" /></td>
    </tr>
    <tr>
      <th align="center">ToMe <a href="#ref-5">[5]</a> - 32% Pruned</th>
      <th align="center">Importance-based <a href="#ref-6">[6]</a> - 32% Pruned</th>
      <th align="center">IDF <a href="#ref-7">[7]</a> - 32% Pruned</th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/other_training_free_pruning/tome/penguin.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/other_training_free_pruning/importance_based/penguin.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/other_training_free_pruning/idf/penguin.mp4" /></td>
    </tr>
  </table>

## Visualizations with Time-time-move Integration

- ***Gardening***

  <table>
    <tr>
      <th align="center">Motion Prompt</th>
      <th align="center">TTM <a href="#ref-8">[8]</a></th>
      <th align="center">LIF (Ours) - 47% Pruned</th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/ttm/motion_prompt/gardening.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/ttm/ttm/gardening.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/ttm/lif/gardening.mp4" /></td>
    </tr>
  </table>

- ***Owl***

  <table>
    <tr>
      <th align="center">Motion Prompt</th>
      <th align="center">TTM <a href="#ref-8">[8]</a></th>
      <th align="center">LIF (Ours) - 47% Pruned</th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/ttm/motion_prompt/owl.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/ttm/ttm/owl.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/ttm/lif/owl.mp4" /></td>
    </tr>
  </table>

- ***Cocktail***

  <table>
    <tr>
      <th align="center">Motion Prompt</th>
      <th align="center">TTM <a href="#ref-8">[8]</a></th>
      <th align="center">LIF (Ours) - 34% Pruned</th>
    </tr>
    <tr>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/ttm/motion_prompt/cocktail.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/ttm/ttm/cocktail.mp4" /></td>
      <td align="center"><video controls autoplay muted loop width="320" src="./assets/ttm/lif/cocktail.mp4" /></td>
    </tr>
  </table>



## Reference

- [1] Huang, Xun, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. "Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion." In Advances in Neural Information Processing Systems. 2025. <a name="ref-1" id="ref-1"></a> 
- [2] Liang, Feng, et al. "Looking Backward: Streaming Video-to-Video Translation with Feature Banks." In The Thirteen International Conference on Learning Representations. 2025. <a name="ref-2" id="ref-2"></a> 
- [3] Kodaira, Akio, Chenfeng Xu, Toshiki Hazama, Takanori Yoshimoto, Kohei Ohno, and others. "StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation." In arXiv. 2023. <a name="ref-3" id="ref-3"></a> 
- [4] Zhang, Yabo, Yuxiang Wei, Dongsheng Jiang, XIAOPENG ZHANG, Wangmeng Zuo, and Qi Tian. "Input Video: Training-free Controllable Text-to-video Generation." In The Twelfth International Conference on Learning Representations. 2024. <a name="ref-4" id="ref-4"></a> 
- [5] Bolya, Daniel, and Judy Hoffman. "Token merging for fast stable diffusion." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2023. <a name="ref-5" id="ref-5"></a> 
- [6] Wu, Haoyu, et al. "Importance-based token merging for efficient image and video generation." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025. <a name="ref-6" id="ref-6"></a> 
- [7] Fang, Haipeng, et al. "Attend to Not Attended: Structure-then-Detail Token Merging for Post-training DiT Acceleration." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025. <a name="ref-7" id="ref-7"></a> 
- [8] Singer, Assaf, et al. "Time-to-Move: Training-Free Motion Controlled Video Generation via Dual-Clock Denoising." *arXiv preprint arXiv:2511.08633* (2025). <a name="ref-8" id="ref-8"></a> 
