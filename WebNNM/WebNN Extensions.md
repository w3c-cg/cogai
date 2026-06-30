# WebNN Extensions  

***Whilst WebNN is a new API, it is based upon traditional approaches to neural networks that are by now being surpassed by newer approaches.  What are these and what kind of WebNN extensions are needed?***  

*The challenge is how to convince the W3C WG to take the requirements seriously enough to agree to take on the extensions when they recharter.  I suspect that compelling demos are needed to show the new kinds of models and how they are slow on WebNN compared to proprietary ML libraries.*  

A general challenge for neural networks is to  represent relational roles independently of their fillers and simultaneously specify which roles are bound to which fillers. Some ways to approach this include:  

* Tensor outer products  
* Transformers and self-attention  
* Vector Symbolic Architectures (VSAs) / Hyperdimensional Computing  
* Biological / Spiking Neural Networks: Temporal Synchrony  
* Recurrent networks with non-euclidean manifolds   
  

| Method | Role Independence? | Simultaneous Binding? | Mechanism |
| ----------------------- | ----------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------- |
| Tensor Products (TPRs) | Yes (Explicit vectors) | Yes (Vector addition) | Outer product matrix algebra |
| Transformers | Yes (Contextual/Positional) | Yes (Multi-head parallel processing) | Query-Key-Value attention  |
| VSAs / Hyperdimensional | Yes (Orthogonal vectors) | Yes (Superposition) | Circular convolution or XOR |
| Spiking Networks | Yes (Distinct neural populations) | Yes (Phase-locked firing) | Temporal synchronization |
| RNNs + Manifolds | Yes (Orthogonal geometric axes/subspaces) | Yes (Unique localisation coordinates) | Continuous attractors, low-rank state steering, and curved topological surfaces |


*What kind of neural network models are needed for manifold based approaches?  What kind of extensions to WebNN are needed to make this work well?*  

To implement manifold-based approaches for role-filler binding, the focus shifts away from vanilla, discrete token-matching networks toward neural architectures that natively model **continuous dynamical systems**.  

Running these models efficiently in consumer hardware (such as via a web browser) requires adapting web-native machine learning standards.  

**1. Network Models Needed for Manifold Approaches**  

Manifold-based binding requires architectures capable of shaping, sustaining, and steering low-dimensional surfaces within high-dimensional activation spaces.  

**Continuous Attractor Neural Networks (CANNs)**  

•     **What they do:** CANNs are designed to maintain a continuous family of steady states (a manifold of attractors, like a ring or a torus).  

•     **Role in Binding:** The manifold itself acts as a structured "track" representing the role (e.g., grammatical position). The network uses localized bumps of activity that can smoothly slide along this track to track fillers without collapsing into a single fixed point.  

**Low-Rank Recurrent Neural Networks (Low-Rank RNNs)**  

•     **What they do:** By constraining the recurrent weight matrix $W$ to be low-rank (e.g., $W = \sum_{i=1}^R m_i n_i^T$), the network's dynamics are mathematically forced to unfold on a low-dimensional hyperplane or manifold.  

•     **Role in Binding:** This forces the network to segregate information into distinct subspaces—such as a "role subspace" and a "filler subspace"—while allowing the recurrent loops to smoothly steer the state vector between them.  

**Neural Ordinary Differential Equations (Neural ODEs)**  

•     **What they do:** Instead of defining discrete layer-by-layer transitions ($h_{t+1} = f(h_t)$), Neural ODEs define the *derivative* of the hidden state ($\frac{dh(t)}{dt} = f(h(t), t)$).  

•     **Role in Binding:** This treats the hidden state as a continuous trajectory moving over a curved geometric manifold, making it highly effective at tracking fluid, context-dependent transitions between structural roles.  

**2. Required Extensions to WebNN**  

The **W3C Web Neural Network API (WebNN)** is hardware-agnostic and built around traditional, feedforward execution graphs (like Transformers and CNNs). It handles static multi-dimensional tensors well, but lacks native primitives for continuous geometric and state-dependent recurrent operations.  

To execute manifold-based cognitive models natively on Web GPUs/NPUs, several critical extensions are needed:  

**A. State-Persistent, Custom Cell Recurrence**  

•     **The Current Limitation:** WebNN heavily prioritises feedforward graphs. While basic lstm and gru operators exist, they are closed boxes that do not allow developers to inject custom, state-dependent manifold warping logic between time steps.  

•     **The Extension Needed:** A low-latency **MLRecurrentContext** or an explicit loop execution primitive (loop()) that allows internal hidden states (MLTensor) to remain resident in GPU/NPU memory across iterations without costly round-trips to JavaScript.  

A work around is to use a double-buffered banked tensors. This keeps the data resident on the device (GPU/NPU) without serialising it back to a CPU-side JavaScript array. However, using a manually managed memory bank pattern in JS to pass tokens sequentially still hits a massive performance wall compared to a native loop primitive like loop(). The bottleneck isn't the data transfer; it's the execution control loop.  

**B. Geometric and Non-Euclidean Distance Metrics**  

•     **The Current Limitation:** WebNN operations assume flat, Euclidean tensor spaces. Calculating coordinates on a curved manifold requires building complex, inefficient sub-graphs out of basic element-wise primitives.  

•     **The Extension Needed:** Native operators for differential geometry, such as:  
    •         **riemannianDistance()**: To measure the distance between two state points along a curved manifold. 
    •         **exponentialMap() / logarithmicMap()**: To project vectors from a flat tangent space onto a curved manifold and vice versa. 

**C. Low-Rank Matrix Decompositions (On-the-Fly)**  

•     **The Current Limitation:** WebNN lacks built-in matrix factorisation tools; it expects static weights.  

•     **The Extension Needed:** Primitives for Singular Value Decomposition (**svd()**) or QR decomposition (**qr()**). This would allow the web runtime to dynamically project massive filler-role spaces down to their active low-dimensional manifold representations on the fly, saving substantial memory bandwidth.  

**D. Zero-Copy Interoperability with WebGPU (Via MLTensor)**  

•     **The Status:** As of 2026, WebNN features the MLTensor API for buffer sharing, but it is primarily optimised for chaining static models (like feeding a camera frame to an object detector).  

•     **The Extension Needed:** Full, bidirectional, zero-copy synchronisation between WebNN graphs and **WebGPU compute shaders**. This would allow a developer to handle the complex, non-linear geometric updates of a neural manifold in a custom WebGPU shader, and seamlessly pass the resulting manifold coordinates back into WebNN for standard layer processing.  

*Would extensions to WebNN for manifolds also be suitable for continual learning and memory based approaches that scale better than transformers? *  

Absolutely. The proposed WebNN extensions for neural manifolds align directly with the structural needs of **continual learning (CL)** and alternative **memory-based architectures** designed to surpass Transformers in scaling efficiency.  

Transformers scale quadratically with context length because they compute an explicit attention map across every past token. To beat this, next-generation architectures rely on **fixed-size state spaces that dynamically update**.  

**How Manifold Extensions Unlock Continual Learning**  

In traditional networks, sequential training causes **catastrophic forgetting** because updating parameters to learn Task B warps the entire landscape, crushing the representations for Task A.   

Geometric and manifold approaches treat continual learning as **Shared-Manifold Continuation** or the construction of **Orthogonal Task Manifolds**.  

•     **Subspace Protection (riemannianDistance & low-rank mappings):** Instead of protecting individual synaptic weights (which is what methods like EWC do), a manifold-aware network ensures that when learning a new task, the overall geometry and global curvature of the existing task manifolds are preserved.  

•     **Zero-Interference Learning:** By using low-rank decompositions, the network can restrict weight updates to the "tangent spaces" orthogonal to the manifolds of previously learned tasks. The network learns Task B in an entirely different geometric direction, completely eliminating forgetting without needing to store or replay massive datasets.  

**Powering Memory-Based Models That Outscale Transformers**  

Several architectures are vying to replace Transformers by compressing context into a highly efficient, recurrently updated mathematical state. The precise WebNN extensions needed for manifolds perfectly accelerate these models:  

**1. State-Space Models (SSMs) like Mamba**  

•     **The Scaling:** linear scaling.  

•     **The Manifold Connection:** SSMs track context by moving a continuous hidden state through a linear dynamical system: dx(t)/dt = Ax(t) + Bu(t)  

•     **WebNN Fit:** The **MLRecurrentContext** loop primitive is exactly what is needed to run SSM scans at hardware speeds in-browser. Without it, the sequential step-by-step updates of the SSM hidden state are throttled by the WebNN-to-JavaScript abstraction layer.  

**2. Linear Attention / Linear Transformers (e.g., RWKV, GLA)**  

•     **The Scaling:** linear inference memory scaling.  

•     **The Manifold Connection:** Instead of a $T \times T$ attention matrix, these architectures maintain a constantly updating context matrix ($M_t = M_{t-1} + V_t K_t^T$), which acts as a dynamic memory manifold.  

•     **WebNN Fit:** Low-rank operators and efficient matrix updating tools let these networks compress entire chat histories into a compact tensor manifold directly on consumer GPUs.  

**3. Hyperdimensional Computing / Vector Symbolic Architectures (VSAs)**  

•     **The Scaling:** linear search and retrieval times.  

•     **The Manifold Connection:** VSAs bind concepts using high-dimensional random vectors that form a hyper-spherical manifold.  

•     **WebNN Fit:** To retrieve a memory without decoding everything, you calculate the geometric distance between vectors. Native **riemannianDistance()** or optimized dot-product distance operators on the WebNN side would allow a browser-based AI to query massive episodic memories instantly, bypassed by the heavy matrix multiplication required by Transformer attention layers.  

**Summary**  

The current WebNN specification is heavily biased toward the **"static, feedforward graph"** paradigm of early-generation deep learning and standard Transformers.  

By expanding WebNN to support **recurrent state persistence**, **non-Euclidean geometry**, and **on-the-fly low-rank projections**, the web standards body wouldn't just be enabling abstract cognitive modelling—they would be providing the infrastructure for highly efficient, linear-scaling, browser-native AI models that can learn continuously from user interaction without melting the device's battery.  

*A cognitively inspired approach would combine short and long term memory. Here short term memory is used in place of a long context window, and complemented by a long term memory which the model queries as part of the chain of thought.  A significant challenge is to incrementally update long term memory without pausing work on the current task.*  

This is exactly where the intersection of **cognitive architecture** and **dynamic neural manifolds** becomes highly practical.  

By mirroring the biological division between the **prefrontal cortex** (working memory/short-term manifold) and the **hippocampus/cortex** (long-term memory manifold), we can bypass the massive computational bottleneck of the Transformer's context window.  

The core challenge you highlighted—**asynchronous, non-blocking updates to long-term memory**—is a fundamental engineering hurdle. If the model has to stop and re-index its entire long-term memory vault every time a new thought occurs, the user experience falls apart.  

Here is how a manifold-based approach solves this challenge, and how it maps to WebNN and modern hardware.  

**1. The Architecture: Dual-Manifold Interaction**  

Instead of one massive context window, the model utilizes two distinct geometric spaces that interact during the Chain of Thought (CoT):  


```
                       [ Input / Context ]
                               │
                               ▼
 ┌───────────────────────────────────────────────────────────┐
 │ Short-Term Working Memory (Dynamic Low-Rank Manifold)     │
 │ • Holds current task context & active CoT tokens          │
 │ • Fast, continuous, fluid trajectories (Neural ODE / SSM) │
 └─────────────┬───────────────────────────────▲─────────────┘
               │                               │
    Asynchronous Projection           Retrieval Query
  (Low-Rank Synaptic Tagging)       (Manifold Geometry)
               │                               │
 ┌─────────────▼───────────────────────────────┴─────────────┐
 │ Long-Term Episodic Memory (High-Dimensional Hyper-Manifold)│
 │ • Consolidated structural knowledge & historical facts    │
 │ • Static or slowly evolving topological surfaces          │
 └───────────────────────────────────────────────────────────┘

```


**Short-Term Working Memory (STWM)**  

•     **Mechanism:** A highly dynamic, low-rank recurrent manifold (like a Mamba state-space or an RNN trajectory).  

•     **Role:** It acts as the active "canvas" for the Chain of Thought. It doesn't store every word verbatim; it stores the *geometric trajectory* of the current reasoning path.  

**Long-Term Memory (LTM)**  

•     **Mechanism:** A vast, high-dimensional, highly structured hyper-manifold (such as a Vector Symbolic Architecture or an orthogonal attractor landscape).  

•     **Role:** A permanent repository. When the CoT encounters a gap, the STWM coordinates project a query vector into the LTM manifold, instantly pulling relevant historical contexts back into the active working subspace.  

**2. Solving the Challenge: Incremental, Background Memory Updates**  

To update the long-term memory without pausing active generation, the system needs to separate the **immediate representation** of a memory from its **global structural consolidation**.  

**Phase 1: Synaptic Tagging and Localised Injections (Immediate & Non-Blocking)**  

When the model completes a step in its Chain of Thought that is deemed worthy of long-term storage, it doesn't immediately recompute the entire long-term manifold. Instead, it uses **Synaptic Tagging**—a concept borrowed from neuroscience.  

•     The new memory is converted into a low-rank outer product ($\Delta W = f \otimes r$).  

•     This tiny, isolated vector matrix is appended to a local "buffer zone" on the edge of the long-term memory manifold.  

•     **The Benefit:** This takes negligible compute and happens in parallel with the next token generation step. The model's active generation never stutters.  

**Phase 2: Asynchronous Geometric Relaxation (Background Consolidation)**  

While the model continues working on the current task using its short-term manifold, a background process gradually integrates the new memory buffer into the main long-term hyper-manifold.  

•     **Manifold Relaxation:** Think of the long-term memory manifold as a rubber sheet. A new memory is like dropping a tiny pebble onto it. Instead of violently reshaping the sheet instantly, the network uses a background thread to let the sheet smoothly deform and "relax" around the new point.  

•     **Orthogonal Embedding:** Using the low-rank properties discussed earlier, the background update is strictly constrained to directions that are *orthogonal* to the manifold regions currently being read by the short-term working memory.  

**3. WebNN Infrastructure Needed for Asynchronous CoT Memory**  

To make this dual-manifold, background-updating memory system work smoothly in a web browser, WebNN requires specific implementation paradigms that borrow heavily from modern graphics pipeline architectures:  

**A. Non-Blocking Graph Execution & Multi-Threading**  

•     **The Need:** Standard WebNN executes command queues sequentially. If a background memory consolidation graph is submitted, it shouldn't block the high-priority, low-latency token generation graph.  

•     **The Extension:** WebNN needs explicit support for **Asynchronous Execution Queues** (similar to computing queues in Vulkan/WebGPU). Developers must be able to flag the Short-Term CoT graph as Priority: Realtime and the Long-Term Memory update graph as Priority: Background, allowing the hardware's scheduler to interleave them seamlessly.  

**B. Double-Buffered State Matrices (**MLTensor** Swapping)**  

•     **The Need:** While the background thread is gradually updating the long-term memory manifold, the active CoT model still needs to query it without encountering race conditions or memory corruption.  

•     **The Extension:** A native **double-buffering mechanism** for MLTensor. The model queries LTM_Buffer_A while the background consolidation process safely writes the relaxed, updated geometric coordinates to LTM_Buffer_B. Once the update is complete, a zero-copy pointer swap occurs.  

**C. Gated Hebbian Plasticity Operators**  

•     **The Need:** To write to the temporary long-term buffer instantly without heavy back propagation gradients.  

•     **The Extension:** Simple, hardware-accelerated learning primitives like **outerProductUpdate(Matrix, VectorA, VectorB, LearningRate)**. This allows the short-term working memory to "stamp" an echo of its current state directly into the long-term memory buffer using rapid, local Hebbian updates right on the NPU/GPU, entirely bypassing the need for a standard optimisation loop.  

*Which modern ML frameworks support the extensions discussed above?*  

To implement the geometric, recurrent, and asynchronous long-term memory architectures we’ve discussed, standard high-level ML frameworks (like vanilla ONNX Runtime, Hugging Face, or basic TensorFlow) actually struggle out of the box. They suffer from the same "kernel fragmentation" and JS-loop overheads we just covered.  

However, several cutting-edge frameworks and lower-level deep learning systems have specifically introduced features to support these low-rank, manifold-based, and state-persistent architectures.  

**1. PyTorch & Triton (The Gold Standard for Fused RNN/SSM Operators)**  

PyTorch does not natively loop states efficiently via pure Python code, but it provides the infrastructure to build the exact **fused custom kernels** required for manifold state-steering.  

•     **Triton (by OpenAI):** Triton allows developers to write highly parallelised GPU code in a Python-like language. Frameworks like **Mamba** (mamba-ssm) and Linear Attention models (like GLA / Gated DeltaNet) are written entirely in Triton.  

•     **How it handles the extensions:** Triton bypasses the host framework's graph loop by compiling the entire recurrent scan or low-rank update into a *single, fused GPU kernel*. The hidden state stays entirely within the GPU’s ultra-fast SRAM cache across thousands of tokens, rather than bouncing back and forth to VRAM or the CPU.  

**2. Core ONNX (Recent 2026 Shift)**  

Until recently, trying to export a manifold/state-space model to ONNX resulted in a massive, unrolled graph or a highly inefficient Loop node operator that ruined edge/browser-based performance.  

•     **LinearAttention Operators:** In early 2026, the ONNX steering committee introduced explicit, native proposals to add **LinearAttentionRecurrent** and structured state-update primitives natively to the ONNX spec.  

•     **How it handles the extensions:** Rather than letting a compiler break a recurrent step down into 20 fragmented math operations, it treats the continuous state update as a single atomic node. Hardware execution providers (like DirectML or TensorRT) can then map that node to a perfectly optimised, hardware-level hardware block.  

**3. Jax & Equinox (For Differential Geometry & Neural ODEs)**  

When it comes to the complex math required for actual manifold navigation (like computing Riemannian distances, non-Euclidean vector trajectories, or back propagating through Neural ODEs), **JAX** is the undisputed champion.  

•     **Just-In-Time (JIT) Compilation:** JAX uses XLA (Accelerated Linear Algebra) to compile entire complex math functions into single fused kernels.  

•     **Equinox / Diffrax:** Built on top of JAX, libraries like Diffrax allow you to define and execute Neural ODEs natively. It treats hidden state propagation as a continuous geometric trajectory, perfectly executing the "manifold relaxation" and "state steering" concepts natively on the GPU without intermediate memory overhead.  

**4. Apple CoreML (Advanced State-Persistence on Edge Hardware)**  

For consumer/edge devices, Apple’s **CoreML** actually sets a massive precedent that WebNN should follow.  

•     **Stateful Subgraphs (MLState):** CoreML has explicit support for stateful tensors. When deploying a model (like an Apple-optimized Mamba or an RNN), you can define a persistent tensor that remains resident on the Apple Neural Engine (ANE) between independent execution requests. The CPU just passes the next input token, and the ANE runs the internal recurrences completely sealed from the host OS.  

**Summary of Framework Support**  


| Framework / Ecosystem | Feature Matrix | Suitability for Manifolds & Advanced Memory |
| --------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------- |
| PyTorch + Triton | Fused hardware kernels, custom memory layouts | Excellent for training and high-end backend inference. |
| JAX + Diffrax | Native differential geometry, ODE solvers, XLA fusion | Excellent for theoretical modelling of geometric landscapes. |
| CoreML | Native MLState support on NPUs | Excellent for edge-based, state-persistent deployments. |
| ONNX Runtime | Native Loop (slow); New LinearAttention spec | Improving as it moves away from fragmented graphs to native SSM blocks. |
| WebNN (Current) | Static feedforward graphs only | Poor (Requires the exact extensions we discussed to catch up to CoreML/Triton). |

## Experimental Approach

To motivate improvements to WebNN, I could demonstrate extensions to my model syntax and DAG that run on the native NPU/GPU framework for MacOS, i.e. CoreML and Metal shaders.

I would extend my DAG node classes to export to Apple's MIL JSON-based format, and then  use Apple's native command-line tool `xcrun coremlcompiler compile` within the Node ecosystem to generate the binary `.mlmodelc` artifact without touching Python.  Gemini suggests creating a Rust-based bridge from Node to the CoreML runtime, and combining CoreML with Metal Shaders for custom operations, where the bridge enables shared memory pointers without dropping down to CPU RAM.  This would enable complex, non-standard loss functions (such as optimizing for *Riemannian distance* along a curved manifold surface), Asynchronous Geometric Relaxation and rapid local Hebbian updates ($W = f \otimes r$).

I may also want to explore using Linux and the emerging GPUs and NPUs for smartphones, laptops and chromebooks. Ideally, this would be on widely available devices so others can easily run my applications and take them further. For this, I may want to consider using JAX.

JAX is a Python-based functional programming framework as distinct from PyTorch which uses an object oriented approach. JAX uses   Google's XLA compiler to trace Python code, optimize the mathematical graph, fuse operations together, and compile it directly into specialized GPU or TPU machine code.
