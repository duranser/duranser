# Loop Tiling

Loop tiling is a technique used to improve
data locality and data reuse by dividing a large loop iteration space into smaller
blocks, called tiles. For CNN layers, the data required to process the complete
iteration space may exceed the available on-chip memory capacity. Loop tiling
addresses this limitation by partitioning the computation into smaller tiles, allowing
the corresponding input activations, weights, and partial sums to be managed within
the available on-chip memory.

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/755b2ce3-a052-408d-81c6-14604883df1d"
    alt="Output-channel Parallelism"
    width="400"
  />
</p>

<p align="center"><em>Figure 1. Loop Tiling in FPGAs.</em></p>

In FPGA-based CNN designs, input activations and weights are transferred from external memory 
to dedicated input and weight tile buffers implemented using BRAM resources. 
These buffers provide the PE array with the data required for convolution operations. 
The PE array performs MAC operations using dedicated DS slices and/or LUT resources. 
The resulting partial sums are stored in an on-chip buffer implemented using registers or BRAMs. 
By keeping input activations, weights, and partial sums on-chip, 
the architecture enables data reuse across successive computations and reduces external memory traffic.


## Output-Channel Tiling

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/fd7ea610-0229-485b-9442-d019faedc454"
    alt="Output-channel Tiling"
    width="400"
  />
</p>

<p align="center"><em>Figure 2. Output-channel Tiling.</em></p>

Since the same input activations contribute to multiple output channels, retaining the input data
in on-chip buffers while processing successive output-channel tiles promotes activation
reuse and reduces external memory accesses. The tile size of the output-channel is $T_M$. 
$M$ denotes the total number of output channels, and $K = \lceil M/T_M \rceil$ represents
the number of output-channel tiles. All $K$ tiles share the same input activations.


## Input-Channel Tiling

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/3cfefe1d-3a89-4850-96f5-24e5a7efa99b"
    alt="Input-channel Tiling"
    width="400"
  />
</p>

<p align="center"><em>Figure 3. Input-channel Tiling.</em></p>

Input-channel tiling allows the convolution’s accumulation operation to be performed incrementally. 
The tile size of the input-channel is $T_M$. 
$N$ denotes the total number of input channels, and $C = \lceil N/T_C \rceil$ represents
the number of input-channel tiles. All $C$ tiles share the same input activations.


## Spatial Tiling

Spatial tiling partitions the spatial dimensions (height and width) of a convolutional
layer into smaller spatial tiles. In a convolution, neighboring output positions have
overlapping receptive fields. Therefore, input activations can be reused across multiple
convolution operations when the required input region is retained in local memory.

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/7282477d-edea-48c9-bfe5-63f40056da53"
    alt="Spatial Tiling"
    width="500"
  />
</p>

<p align="center"><em>Figure 4. Spatial Tiling.</em></p>

The example of the figure above is an example for a convolutional layer 
with a $3 \times 3$ kernel and a stride of 1. The $8 \times 8$ output feature map 
is divided into four $4 \times 4$ tiles. Each output tile requires a corresponding $6 \times 6$ 
input region, including the boundary pixels. 

Consequently, the input regions associated with 
neighboring output tiles overlap, allowing certain input activations to be reused across tile computations. 
Retaining these overlapping regions in on-chip buffers can reduce redundant external memory accesses.


## References

[1] Zhang, C., Li, P., Sun, G., Guan, Y., Xiao, B. and Cong, J. (2015). Optimizing FPGA-Based Accelerator Design for Deep Convolutional Neural Networks, *Proceedings of the ACM/SIGDA International Symposium on Field-Programmable Gate Arrays (FPGA)*, Monterey, CA, USA, pp. 161–170.

[2] Ma, Y., Cao, Y., Vrudhula, S. and sun Seo, J. (2018). Optimizing the Convolution Operation to Accelerate Deep Neural Networks on FPGA, *IEEE Transactions on Very Large Scale Integration (VLSI) Systems*, 26(7), 1354–1367.

