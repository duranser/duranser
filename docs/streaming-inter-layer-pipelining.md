## Streaming

Streaming is a data-processing approach in which input activations are supplied to the accelerator incrementally and processed as they become available, 
rather than requiring the entire input feature map to be loaded into local memory before computation begins.

Within a convolutional layer, streaming allows intermediate results to be forwarded between processing stages without requiring complete feature maps to be stored between these stages. 
Therefore, streaming reduces intermediate buffering requirements and supports continuous processing of incoming data [1-3].

Common hardware components supporting streaming include:

- **Line buffers:** Store the input rows required to construct convolution windows.

- **Sliding-window generators:** Supply overlapping convolution windows to the processing elements.

- **FIFOs:** Temporarily buffer data between processing stages and accommodate differences in their production and consumption rates.

- **Streaming interfaces:** Transfer data between hardware modules using protocols such as AXI4-Stream.

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/035018c8-cef5-4c3d-b6a6-a25105ce730f"
    alt="Streaming architecture"
    width="900"
  />
</p>

<p align="center"><em>Figure 1. Streaming architecture for a convolutional layer.</em></p>

The figure above shows a streaming architecture in a CNN accelerator:

- Input activations are transferred incrementally from external memory through a streaming interface, or a preceding convolutional layer with a FIFO buffer. 

- Line buffers store incoming activations to retain the input rows required for convolution window formation.
  
- A sliding-window generator constructs overlapping convolution windows.

- The PE array performs MAC operations, then the resulting values are subsequently processed through bias addition and a nonlinear activation function.

- The generated output activations are then forwarded incrementally to external memory through a streaming interface or a subsequent convolutional layer with a FIFO buffer.



## Inter-Layer Pipelining

Inter-layer pipelining extends incremental data processing across consecutive CNN layers by allowing their execution to overlap. 
A downstream layer begins processing once sufficient intermediate activations become available from the preceding layer, 
rather than waiting for the entire intermediate feature map to be generated [1-3]. 

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/497ab3a3-461a-449f-a599-e19fd868cb4f"
    alt="Inter-layer pipelinig"
    width="800"
  />
</p>

<p align="center"><em>Figure 2. Inter-layer pipelinig.</em></p>

The figure above illustrates an inter-layer pipelinig example. Each engine performs the convolution and activation operations associated with one CNN layer. 
FIFO buffers connect neighboring engines and temporarily store intermediate feature-map data. As Layer 1 generates output activations, 
they are transferred through FIFO 1 to Layer 2.
The same relationship is established between Layers 2 and 3 and between Layers 3 and 4, allowing multiple layers to execute concurrently. 

Inter-layer pipelining reduces the need to store complete intermediate feature maps. 
Consequently, intermediate activations can remain on-chip and flow directly through the layer pipeline instead of being written to and subsequently retrieved from external memory.

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/68497a6b-f76b-4cff-a7c2-ed7ff38be876"
    alt="pipeline schedule"
    width="500"
  />
</p>

<p align="center"><em>Figure 3. Balanced pipeline schedule.</em></p>

Each layer engine processes one tile during a single pipeline interval, resulting in an initiation interval of one pipeline interval.
During interval $t_3$, all four layer engines operate simultaneously: Layer 1 processes Tile 3, Layer 2 processes Tile 2, Layer 3 processes Tile 1, and Layer 4 processes Tile 0.

After the initial pipeline-filling period, one completed output tile is produced during each successive interval until all four tiles have been processed.
The end-to-end latency of Tile 0 is four pipeline intervals, as it must pass through all four layer engines.

The ideal schedule requires the processing rates of the layer engines to be balanced. 
The slowest pipeline stage limits the sustained throughput of the complete pipeline. Faster upstream engines may experience backpressure, 
while faster downstream engines may remain idle due to insufficient input data. Layer-specific parallelism factors, PE-array dimensions, 
and buffer bandwidths should therefore be selected to balance the effective processing rates as closely as possible.


## References

[1] Zhang, C., Li, P., Sun, G., Guan, Y., Xiao, B., and Cong, J. (2015). "Optimizing FPGA-Based Accelerator Design for Deep Convolutional Neural Networks." *Proceedings of the ACM/SIGDA International Symposium on Field-Programmable Gate Arrays (FPGA)*, Monterey, CA, USA, pp. 161–170.

[2] Ma, Y., Cao, Y., Vrudhula, S., and Seo, J.S. (2018). "Optimizing the Convolution Operation to Accelerate Deep Neural Networks on FPGA." *IEEE Transactions on Very Large Scale Integration (VLSI) Systems*, 26(7), 1354–1367.

[3] Qiu, J., Wang, J., Yao, S., Guo, K., Li, B., Zhou, E., Yu, J., Tang, T., Xu, N., Song, S., Wang, Y., and Yang, H. (2016). "Going Deeper with Embedded FPGA Platform for Convolutional Neural Network." *Proceedings of the ACM/SIGDA International Symposium on Field-Programmable Gate Arrays (FPGA)*, pp. 26–35.

