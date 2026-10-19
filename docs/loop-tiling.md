# Loop Tiling

Loop tiling is a technique used to improve
data locality and data reuse by dividing a large loop iteration space into smaller
blocks, called tiles. 

For CNN layers, the data required to process the complete
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

The PE array performs MAC operations using dedicated DSP slices and/or LUT resources. 
The resulting partial sums are stored in an on-chip buffer implemented using registers or BRAMs. 
By keeping input activations, weights, and partial sums on-chip, 
the architecture enables data reuse across successive computations and reduces external memory traffic.

## Tiling Strategies

A convolution can be tiled along different loop dimensions.

| Strategy | Partitioned dimension | Main consideration |
| --- | --- | --- |
| **Output-channel tiling** | Output channels | Reuse input activations while processing different filter groups. |
| **Input-channel tiling** | Input channels | Accumulate contributions from successive input-channel tiles into the same outputs. |
| **Spatial tiling** | Output height and/or width | Reuse overlapping input regions and filter weights across output positions. |

These strategies are not mutually exclusive, an FPGA accelerator may tile several dimensions simultaneously [1], [2].


## Output-Channel Tiling

Output-channel tiling divides the output channels into smaller groups. Each group is computed using the subset of filter weights associated with its output channels.

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/fd7ea610-0229-485b-9442-d019faedc454"
    alt="Output-channel Tiling"
    width="400"
  />
</p>

<p align="center"><em>Figure 2. Output-channel Tiling.</em></p>

For a layer with $M$ output channels and an output-channel tile size of $T_M$, the number of tiles is

$$
K = \left\lceil \frac{M}{T_M} \right\rceil.
$$

All $K$ tiles share the same input activations.

Different output-channel tiles use different filter weights but operate on the same input feature map. If the required input activations remain in on-chip buffers while successive output-channel tiles are processed, the accelerator can reuse them instead of fetching them repeatedly from external memory.


## Input-Channel Tiling

Input-channel tiling partitions the input channels into smaller groups. Each input-channel group contributes a partial sum to the same output elements.

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/3cfefe1d-3a89-4850-96f5-24e5a7efa99b"
    alt="Input-channel Tiling"
    width="400"
  />
</p>

<p align="center"><em>Figure 3. Input-channel Tiling.</em></p>

For $N$ input channels and an input-channel tile size of $T_C$, the number of tiles is

$$
C = \left\lceil \frac{C_{\mathrm{in}}}{T_C} \right\rceil.
$$

Each tile contains a different subset of input channels and the corresponding weight channels.

Input-channel tiling allows the convolution’s accumulation operation to be performed incrementally. 
All $C$ tiles contribute to the same output partial sums, which is retained in on-chip during accumulation.


## Spatial Tiling

Spatial tiling partitions the output feature map along its height and/or width. Each output tile is computed from a corresponding region of the input feature map.

Spatial tiling partitions the spatial dimensions (height and width) of a convolutional
layer into smaller spatial tiles. In a convolution, neighboring output positions have
overlapping receptive fields. Therefore, input activations can be reused across multiple
convolution operations when the required input region is retained in local memory.

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/7282477d-edea-48c9-bfe5-63f40056da53"
    alt="Spatial Tiling"
    width="520"
  />
</p>

<p align="center"><em>Figure 4. Spatial Tiling.</em></p>

The example of the figure above is an example for a convolutional layer 
with a $3 \times 3$ kernel and a stride of 1. The $8 \times 8$ output feature map 
is divided into four $4 \times 4$ tiles. Each output tile requires a corresponding $6 \times 6$ 
input region, including the boundary pixels. 

Consequently, the input regions associated with neighboring output tiles overlap, allowing certain input activations to be reused across tile computations. 
Retaining these overlapping regions in on-chip buffers can reduce redundant external memory accesses.


## Tile-Size Considerations

Tile sizes are typically selected by considering the following hardware constraints and performance requirements [1], [2]:

- **On-chip memory capacity:** Input activation, weight, and partial-sum tiles must fit within the available on-chip buffers.
- **Memory bandwidth:** Tile sizes should facilitate data reuse to reduce off-chip memory accesses and sustain the data throughput required by the PE array.
- **PE utilization:** Tile dimensions should map efficiently to the available computational parallelism to minimize idle processing elements.


## References

[1] Zhang, C., Li, P., Sun, G., Guan, Y., Xiao, B. and Cong, J. (2015). Optimizing FPGA-Based Accelerator Design for Deep Convolutional Neural Networks, *Proceedings of the ACM/SIGDA International Symposium on Field-Programmable Gate Arrays (FPGA)*, Monterey, CA, USA, pp. 161–170.

[2] Ma, Y., Cao, Y., Vrudhula, S. and sun Seo, J. (2018). Optimizing the Convolution Operation to Accelerate Deep Neural Networks on FPGA, *IEEE Transactions on Very Large Scale Integration (VLSI) Systems*, 26(7), 1354–1367.

