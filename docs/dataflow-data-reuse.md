# Dataflow

Dataflow determines how input activations, weights, and partial sums move between memories and PEs during computation. 
The primary objective of a dataflow is to maximize data reuse within low-cost local storage 
while minimizing data transfers through higher levels of the memory hierarchy. 

In a stationary dataflow, selected data are kept close to the PEs and reused across multiple operations, 
whereas the remaining data are streamed through the accelerator as needed. 
Since off-chip DRAM accesses typically consume significantly more energy than arithmetic operations or 
accesses to local registers and on-chip buffers, minimizing data movement is essential 
for efficient CNN acceleration [1], [2].


Convolutional layers provide several important forms of data reuse:

- **Weight reuse:** The same filter weights are applied to input activations at multiple spatial positions of the feature map.

- **Input-activation reuse:** An input activation can contribute to several neighboring convolution windows and can also be reused in the computation of multiple output channels.

- **Partial-sum reuse:** A partial sum is repeatedly updated with contributions from different kernel positions and input channels until the corresponding output activation is complete.

Different stationary dataflows emphasize different forms of reuse. 
  
| Dataflow Strategy | Primary Data Reuse | Description |
|---|---|---|
| **Weight-Stationary (WS)** | Weight reuse | Keeps weights close to the PEs for repeated computations. |
| **Output-Stationary (OS)** | Partial-sum reuse | Keeps partial sums locally until accumulation is complete. |
| **Input-Stationary (IS)** | Input-activation reuse | Keeps input activations close to the PEs for repeated computations. |
| **Row-Stationary (RS)** | All three forms | Exploits weight, input-activation, and partial-sum reuse across multiple levels of the memory hierarchy. |


## Weight-Stationary Dataflow

In a weight-stationary dataflow, convolution weights are loaded from the weight buffer into local PE storage and 
retained there for multiple MAC operations. Input activations are streamed or broadcast through the PE array, 
allowing each PE to repeatedly use its locally stored weight while successive convolution windows are processed.

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/7e2526e7-28fe-46a9-a9da-c26ec1e50798"
    alt="Weight-Stationary Dataflow"
    width="800"
  />
</p>

<p align="center"><em>Figure 1. Weight-Stationary Dataflow.</em></p>

Weights $W_0, W_1, \ldots, W_N$ are initially loaded into $PE_0, PE_1, \ldots, PE_N$, and remain stationary 
during the processing of multiple output spatial positions.
For output position $Y$, input activation elements $X_0, X_1, \ldots, X_N$ are supplied to their respective PEs. 

Each PE multiplies its streamed activation by its locally stored weight and adds the result to the incoming partial sum. 
The updated partial sum is forwarded through the next PE until output activation $Y$ is computed. For subsequent 
spatial positions, new input activations are supplied while the same weights remain stored in the PEs.


## Output-Stationary Dataflow

In an output-stationary dataflow, each PE is assigned one or more output activations and retains their 
partial sums in a local accumulator until the convolution reduction is complete. During every processing step, 
the PE receives the required input activation and weight, performs a MAC operation, and updates the same locally 
stored partial sum. This process continues across all input channels and kernel positions contributing 
to the assigned output activation.

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/f622aff1-645d-4c4d-85ea-0f746dbccf8a"
    alt="Output-Stationary Dataflow"
    width="600"
  />
</p>

<p align="center"><em>Figure 2. Output-Stationary Dataflow.</em></p>

The required input activations $X_0, X_1, \ldots, X_N$ and weights $W_0, W_1, \ldots, W_N$ are supplied 
from their respective buffers to the corresponding PEs. Each PE receives the activation and weight operands 
associated with its assigned output and accumulates their product into a locally retained partial sum.

As successive activation and weight operands are processed, the partial sums $P_0, P_1, \ldots, P_N$ remain stationary 
within their corresponding PEs, while their values are updated through successive MAC operations.
The completed outputs are written to the output-activation buffer only after the final reduction step.

## Input-Stationary Dataflow

In an input-stationary dataflow, input activations are loaded into registers or local buffers associated with the PEs 
and retained for multiple MAC operations. While the input activations remain stationary, different convolution weights 
are streamed to the PEs. 

Input-stationary dataflow can exploit two important forms of activation reuse. 
First, the same input activation can be reused across multiple filters when computing different output channels. 
Second, since neighboring convolution windows overlap, an input activation can contribute to multiple spatially 
adjacent output positions. 

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/835faf58-959c-4719-8e18-80198a8939f4"
    alt="Input-Stationary Dataflow"
    width="600"
  />
</p>

<p align="center"><em>Figure 3. Input-Stationary Dataflow.</em></p>

Input activations $X_0, X_1, \ldots, X_N$ are loaded from the input-activation buffer and retained locally in PEs.
During computation, the corresponding weights $W_0, W_1, \ldots, W_N$ are streamed from the weight buffer to the PEs.

Each PE multiplies its locally stored activation by the incoming weight and forwards the resulting product to the accumulator.
Once all required contributions have been accumulated, the corresponding output activations $Y_0, Y_1, \ldots, Y_N$ are transferred to the output-activation buffer.


## Row-Stationary Dataflow

Row-stationary dataflow was introduced in the  Eyeriss accelerator [2] to minimize the total energy 
consumed by data movement in a spatial architecture. Unlike stationary dataflows that primarily optimize the movement 
of one data type, row-stationary dataflow attempts to exploit multiple forms of reuse across the memory hierarchy. 
These include the filter reuse of weights, the input-feature-map reuse of activations, and the local accumulation of 
partial sums.

The Row-stationary dataflow first decomposes the 2D convolution operation into 1D convolution primitives that can be 
processed concurrently. Each 1D convolution primitive processes a row of filter weights and a corresponding row of 
input activations, producing a row of partial sums [2].

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/f51ce800-0844-4cc9-affa-079c35a6cb61"
    alt="Row-Stationary Dataflow 1D Primitives"
    width="550"
  />
</p>

<p align="center"><em>Figure 4. Row-Stationary Dataflow 1D Primitives.</em></p>

As an example, 1D convolution primitive in which one three-element filter row $[w_0, w_1, w_2]$ is applied to a five-element input-feature-map row $[x_0, x_1, x_2, x_3, x_4]$.

The PE evaluates three overlapping windows and generates the following row of partial sums:

$$
\begin{aligned}
psum_0 &= w_0x_0 + w_1x_1 + w_2x_2 \\
psum_1 &= w_0x_1 + w_1x_2 + w_2x_3 \\
psum_2 &= w_0x_2 + w_1x_3 + w_2x_4
\end{aligned}
$$

Each partial sum requires three MAC operations. As the convolution window moves along the input row, the same filter weights are reused at successive spatial positions, 
while overlapping input activations are reused between neighboring windows. The three resulting values form one row of partial sums produced by this 1D primitive.

Row-stationary dataflow organizes multiple logical PEs into a logical PE set that collectively performs a 2D convolution. 
Within this set, filter rows are reused horizontally across PEs, input-feature-map rows are reused diagonally, 
and partial sums are accumulated vertically across the logical PE array. By combining local operand reuse, 
inter-PE data sharing, and partial-sum accumulation, row-stationary dataflow aims to minimize overall data-movement energy 
rather than prioritizing the movement of only one data type [2].

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/5c62a7e4-884c-4f84-9fde-f10b26ab7d9f"
    alt="Row-Stationary Dataflow 2D Organization"
    width="950"
  />
</p>

<p align="center"><em>Figure 5. Row-Stationary Dataflow 2D Organization.</em></p>

In horizontal filter-row reuse, the same filter row is shared among PEs assigned to different output-feature-map rows. 
This sharing prevents identical filter values from being repeatedly retrieved from the global buffer, 
thereby reducing weight movement. 

In diagonal input-activation reuse, overlapping convolution windows allow the 
same input-feature-map row to contribute to multiple 1D primitives associated with different filter rows and output positions.
The required input rows are therefore shared diagonally across the logical PE set. 

In vertical partial-sum accumulation, 
the partial sums generated from different rows of a 2D filter are accumulated across vertically adjacent PEs [2].


## References

[1] Sze, V., Chen, Y.H., Yang, T.J., and Emer, J.S. (2017). "Efficient Processing of Deep Neural Networks: A Tutorial and Survey." *Proceedings of the IEEE*, 105(12), 2295–2329.

[2] Chen, Y.H., Krishna, T., Emer, J.S., and Sze, V. (2017). "Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks." *IEEE Journal of Solid-State Circuits*, 52(1), 127–138.
