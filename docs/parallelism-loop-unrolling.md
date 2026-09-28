## Parallel MAC/PE Array Design

Parallel **Multiply-and-Accumulate (MAC)** and **Processing Element (PE) array** design is one of the most fundamental 
architectural techniques for accelerating **Convolutional Neural Network (CNN)** inference. The main idea is to instantiate multiple MAC
units or PEs so that many convolution operations are performed in parallel. Spatial
architectures based on parallel PEs are particularly suitable for FPGA and ASIC
implementations because computations can be distributed across multiple hardware
units and data can be transferred directly between neighboring or logically connected
PEs [1].

A PE may implement one MAC per cycle, several MACs per cycle, or even an entire
small dot product. A PE is the basic computational unit of a CNN accelerator. A
typical PE contains:

- one or more multipliers,
- an adder or accumulator,
- local registers,
- optional local buffers for weights, activations, or partial sums.

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/21e453ae-6f90-4bc2-87da-8993b62d7db7"
    alt="Processing element with three multipliers"
    width="500"
  />
</p>

<p align="center"><em>Figure 1. A processing element (PE) with three multipliers.</em></p>

A simple PE performs:

$$
\mathrm{psum} \leftarrow \mathrm{psum} + \mathrm{input} \times \mathrm{weight}
$$

PEs can be placed in a **1D array** or **2D array**. Both 1D and 2D PE arrays can execute standard 2D convolutions. 
Their main difference is how convolution loops are mapped onto the hardware and how weights,
activations, and partial sums move between PEs.

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/82c044c9-f27a-44ba-8e94-b376824e0d3b"
    alt="1D PE array"
    width="480"
  />
</p>

<p align="center"><em>Figure 2. 1D PE array.</em></p>

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/0cf282d6-30df-4ba4-bd13-285239241dfb"
    alt="2D PE array"
    width="430"
  />
</p>

<p align="center"><em>Figure 3. 2D PE array.</em></p>


| Organization | Illustrative mapping | Main design concern |
|---|---|---|
| **1D array** | A linear sequence of PEs processes different channels, spatial positions, or partial sums. | Efficient operand delivery and data reuse along the PE chain. |
| **2D array** | One PE-array dimension can map to output channels, while the other maps to output spatial positions. | Operand distribution across both array dimensions and efficient routing of partial sums or outputs. |



## Loop Unrolling

A standard convolutional layer, assuming unit stride and no padding, can be represented by the following simplified sequential loop nest:

```text
for m in output_channels:
    for h in output_height:
        for w in output_width:
            psum = 0

            for c in input_channels:
                for r in kernel_height:
                    for s in kernel_width:
                        psum += W[m][c][r][s] * X[c][h+r][w+s]

            Y[m][h][w] = psum
```

The sequential loop nest is a direct computational implementation of the mathematical convolution operation:

```math
Y_m(h,w)
=
\sum_{c=0}^{C_{\mathrm{in}}-1}
\sum_{r=0}^{K_h-1}
\sum_{s=0}^{K_w-1}
W_{m,c}(r,s)\,
X_c(h+r,w+s)
```

In the loop-based representation, the outer loops over `m`, `h`, and `w` enumerate all output channels and output spatial positions. 
Therefore, each combination of `(m,h,w)` corresponds to the computation of one output element $Y_m(h,w)$.

For each output element, `psum` is initialized to zero. The inner loops over `c`, `r`, and `s` directly implement the three summations in the convolution equation. 
At each iteration of the innermost loop, one weight and one input activation are multiplied and accumulated into the partial sum:

```math
\mathrm{psum}
\leftarrow
\mathrm{psum}
+
W_{m,c}(r,s)\,
X_c(h+r,w+s)
```

After all input channels and kernel positions have been processed, `psum` contains the complete value of the corresponding output element and is assigned to the output feature map.

**Loop unrolling** is a hardware-oriented transformation that allows multiple iterations
of a loop to be executed concurrently. Instead of reusing a single arithmetic unit
for all iterations, the unrolled loop instantiates multiple computational paths that
operate on different data elements in parallel. The number of concurrently executed
iterations is determined by the unrolling factor [2], [3].

| Unrolled loop | Parallelism type | Result of unrolling | Hardware implication |
|---|---|---|---|
| `m` — output channel | Output-channel parallelism | Output channels are computed concurrently. | Independent outputs can be mapped directly to parallel PEs. |
| `h` — output height | Spatial parallelism | Output spatial positions are computed concurrently along the height dimension. | Independent outputs can be mapped directly to parallel PEs. |
| `w` — output width | Spatial parallelism | Output spatial positions are computed concurrently along the width dimension. | Independent outputs can be mapped directly to parallel PEs. |
| `c` — input channel | Input-channel parallelism | Partial products contributing to the same output activation are generated concurrently. | Requires an adder tree or accumulation unit to combine partial results. |
| `r` — kernel height | Kernel parallelism | Kernel-row contributions to the same output activation are generated concurrently. | Requires an adder tree or accumulation unit to combine partial results. |
| `s` — kernel width | Kernel parallelism | Kernel-column contributions to the same output activation are generated concurrently. | Requires an adder tree or accumulation unit to combine partial results. |


## 1. Output-Channel Parallelism

For a fixed spatial position `(h,w)`, changing the output-channel index $m$ changes the kernel weights `W_{m,c}` and the bias `b_m`, 
while the input activations `Xc` remain the same. Therefore, the computations associated with different output channels are independent and can be performed concurrently.

Output-channel parallelism is achieved by **unrolling the output-channel loop** and assigning different output channels to parallel PEs [2], [3]. 

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/82bf0e44-c1f6-41c7-91c4-87612227f03c"
    alt="Output-channel Parallelism"
    width="430"
  />
</p>

<p align="center"><em>Figure 4. Output-channel Parallelism.</em></p>

Each PE receives the same input activation window but operates on the weights associated with a different output channel. 
Thus, the input data can be broadcast to all parallel PEs, whereas each PE obtains its own set of kernel weights from the corresponding weight buffer. 
Each PE also maintains an independent partial-sum accumulator. After the accumulation for an output value is completed, the corresponding bias is added and the activation function is applied.

## 2. Spatial Parallelism

Spatial parallelism exploits the independence among different spatial positions of an output feature map. 
It is achieved by **unrolling the output-height and/or output-width loops** so that multiple output pixels are computed concurrently by parallel PEs [2], [3].

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/101b66cc-a417-457c-ab4d-80b686b04f34"
    alt="Output-channel Parallelism"
    width="430"
  />
</p>

<p align="center"><em>Figure 5. Spatial Parallelism.</em></p>

Each PE operates on a different convolution window and generates an output activation at a different spatial location. For a fixed output channel, 
the same filter weights can be shared or broadcast to all spatial PEs, while each PE receives the input activations belonging to its corresponding window. 
Since the generated output pixels are independent, no reduction operation is required between the spatial PEs.

In practice, neighboring convolution windows overlap and share many input activations. 
Therefore, spatial parallelism is commonly supported by **line buffers** and **window buffers**, which reuse previously received input data 
and construct overlapping windows efficiently [4].


## 3. Input-Channel Parallelism

Unlike output-channel parallelism, input-channel parallelism operates along a **reduction dimension**. 
For a given output channel and spatial position, contributions from different input channels belong to the same output activation and must therefore be accumulated.

Input-channel parallelism is achieved by **unrolling the input-channel loop** and assigning different input channels to parallel PEs [2], [3]. 

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/b7a25d5c-591b-43e0-80bd-b92a93e7467e"
    alt="Output-channel Parallelism"
    width="430"
  />
</p>

<p align="center"><em>Figure 6. Input-channel Parallelism.</em></p>

Each PE receives activations from a different input channel together with the corresponding kernel weights and computes a partial sum. Since these partial sums contribute to the same output activation, 
they must be combined using an adder tree or an accumulation unit. After all input-channel contributions have been accumulated, the corresponding bias is added and the activation function is applied.



## 4. Kernel Parallelism

Kernel parallelism exploits the independence among multiplications associated with different spatial positions of the convolution kernel. 
It is achieved by **unrolling the kernel-height and kernel-width loops** so that multiple kernel elements are processed concurrently by parallel PEs [2], [3].

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/a7d2edbf-346c-4482-8248-c704a06ba1ee"
    alt="Output-channel Parallelism"
    width="430"
  />
</p>

<p align="center"><em>Figure 7. Kernel Parallelism.</em></p>

Each PE receives an activation from a different position of the current input window together with the corresponding kernel weight. 
The PE performs a multiplication and produces a partial product. Since these partial products contribute to the same output activation, 
they must be combined using an adder tree or accumulation unit. After all kernel and input-channel contributions have been accumulated, 
the corresponding bias is added and the activation function is applied.


## References

[1] V. Sze, Y.-H. Chen, T.-J. Yang, and J. S. Emer, “Efficient Processing of Deep Neural Networks: A Tutorial and Survey,” *Proceedings of the IEEE*, vol. 105, no. 12, pp. 2295–2329, 2017.

[2] C. Zhang, P. Li, G. Sun, Y. Guan, B. Xiao, and J. Cong, “Optimizing FPGA-Based Accelerator Design for Deep Convolutional Neural Networks,” in *Proceedings of the ACM/SIGDA International Symposium on Field-Programmable Gate Arrays (FPGA)*, Monterey, CA, USA, pp. 161–170, 2015.

[3] Y. Ma, Y. Cao, S. Vrudhula, and J. Seo, “Optimizing the Convolution Operation to Accelerate Deep Neural Networks on FPGA,” *IEEE Transactions on Very Large Scale Integration (VLSI) Systems*, vol. 26, no. 7, pp. 1354–1367, 2018.

[4] J. Qiu, J. Wang, S. Yao, K. Guo, B. Li, E. Zhou, J. Yu, T. Tang, N. Xu, S. Song, Y. Wang, and H. Yang, “Going Deeper with Embedded FPGA Platform for Convolutional Neural Network,” in *Proceedings of the ACM/SIGDA International Symposium on Field-Programmable Gate Arrays (FPGA)*, pp. 26–35, 2016.







