# Loop Pipelining

Loop pipelining is a hardware scheduling technique used to increase throughput by **overlapping consecutive loop iterations in time**. 
Instead of waiting for iteration `i` to complete before starting iteration `i + 1`, 
different operations from multiple iterations are executed concurrently in separate pipeline stages [1].

Consider a loop body composed of four operations:

1. **L** — Load data
2. **M** — Multiply
3. **A** — Accumulate
4. **W** — Write the result

Assume each stage requires one clock cycle. Without pipelining, a new iteration begins only after the previous iteration completes all four stages. However, With loop pipelining, the next iteration can begin before the previous one finishes.

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/f19f877d-c16a-4a71-a868-f291a0b71d8f"
    alt="Processing element with three multipliers"
    width="750"
  />
</p>


The individual iteration still has a latency of 4 cycles, but the pipelined datapath can accept a new iteration every cycle and the loop therefore completes in **7 cycles instead of 16 cycles**.

The key point is that pipelining improves **throughput**, not the **latency** of an individual iteration.



### Latency \(L\)

The **latency** is the number of clock cycles required for one iteration to travel from the beginning to the end of the pipeline.

For the four-stage example above:

$$
L = 4 \text{ cycles}
$$

### Initiation Interval \(II\)

The **initiation interval (II)** is the number of clock cycles between the start of two consecutive loop iterations [1].

- `II = 1` → a new iteration starts every clock cycle.
- `II = 2` → a new iteration starts every two clock cycles.
- Larger values of `II` reduce throughput.

For a loop containing $N_{\text{iter}}$ iterations, the execution time of a pipelined loop is:

$$
C_{pipeline} = L + (N_{iter}-1)\times II
$$

where:

- $C_{\text{pipeline}}$ is the total number of clock cycles,
- $L$ is the pipeline latency,
- $N_{\text{iter}}$ is the number of loop iterations,
- $II$ is the initiation interval.

For the example above, the execution time can be calculated as

$$
C_{\text{pipeline}} = L + (N_{\text{iter}} - 1)\times II = 4 + (4 - 1)\times 1 = 7 \text{ clock cycles}
$$

Without pipelining, the four iterations are executed sequentially and require:

$$
C_{\text{sequential}} = N_{\text{iter}} \times L
= 4 \times 4
= 16 \text{ clock cycles}.
$$

Thus, in this example, loop pipelining reduces the total execution time from 16 clock cycles to 7 clock cycles.

## Throughput & Pipeline Hazards

An initiation interval of 1 is often the desired target in hardware accelerators to achieve high throughput.

If the clock frequency is $f_{clk}$, then the throughput can be represented as

$$
T_{throughput} = \frac{f_{clk}}{II}
$$

iterations per second.

Thus, an $II$ of 1 provides the maximum iteration rate for a given clock
frequency. Increasing the initiation interval reduces the rate at which new
iterations can enter the pipeline.

However, achieving $II = 1$ is not always possible. The achievable initiation interval depends on the structure of the loop and 
the underlying hardware architecture. Loop-carried dependencies, memory-access conflicts,
resource conflicts, and irregular control flow may introduce pipeline hazards or stalls,
resulting in a larger initiation interval.


## 1. Loop-Carried Dependencies

A loop-carried dependency occurs when one iteration depends on a result produced by a previous iteration.

For example:

```cpp
for (int i = 0; i < N; i++) {
    sum = sum + x[i];
}
```

Iteration `i + 1` requires the updated value of `sum` produced by iteration `i`. The feedback path can limit the minimum achievable initiation interval, especially if the accumulation operator requires multiple cycles.

## 2. Memory-Port Limitations

A pipelined loop may require several memory accesses in the same cycle. If the memory architecture cannot provide enough concurrent accesses, the conflicting accesses must be distributed across multiple cycles, increasing the minimum achievable initiation interval.

Typical examples include:

- multiple accesses to the same BRAM,
- simultaneous read and write operations,
- insufficient memory banking,
- conflicting array accesses.


## 3. Resource Conflicts

If multiple operations require the same arithmetic resource during overlapping iterations, they cannot execute concurrently unless additional hardware resources are provided. Therefore, the synthesis tool may schedule them across different cycles.

Examples include shared:

- multipliers,
- DSP blocks,
- dividers,
- floating-point operators.



## 4. Control Flow

Conditional statements inside a loop may cause different iterations to follow different execution paths. 
In a hardware implementation, these paths are typically represented by conditional datapaths, multiplexers, enable signals, and associated control logic [1].

```cpp
for (int i = 0; i < N; i++) {
    if (x[i] > threshold)
        y[i] = f(x[i]);
    else
        y[i] = g(x[i]);
}
```

Such control flow does not necessarily prevent the loop from being pipelined. However, complex conditional paths can increase multiplexing, routing,
control-logic, and resource requirements. If the alternative paths have different latencies, memory-access patterns, or resource requirements, they may
also increase the minimum achievable initiation interval.



## Loop Pipelining vs. Loop Unrolling

Loop pipelining and loop unrolling exploit different forms of parallelism.

### Loop Pipelining

Loop pipelining provides **temporal parallelism**, in which different loop iterations occupy different stages of the datapath at the same time [1], [2].

```text
Cycle 0: I0-S0
Cycle 1: I0-S1 | I1-S0
Cycle 2: I0-S2 | I1-S1 | I2-S0
```

Its primary objective is usually to reduce the initiation interval.

### Loop Unrolling

Loop unrolling provides **spatial parallelism**, in which Multiple operations or iterations are replicated in hardware and executed concurrently [1], [2].

For an unroll factor of four:

```text
Iteration group:

I0 ── Processing Unit 0
I1 ── Processing Unit 1
I2 ── Processing Unit 2
I3 ── Processing Unit 3
```


The two techniques can also be combined. Loop unrolling determines how many operations may execute concurrently within an initiation interval, while loop
pipelining determines how frequently a new group of operations can begin. Their combined use can substantially increase throughput, but also increases
arithmetic-resource usage and memory-bandwidth requirements [2]–[4].

| Technique | Primary objective | Typical benefit | Typical cost |
|---|---|---|---|
| Loop pipelining | Reduce initiation interval | Higher throughput | Registers, control logic, routing |
| Loop unrolling | Increase spatial parallelism | Multiple operations per cycle | DSPs, LUTs, registers, memory ports |
| Pipelining + unrolling | Maximize sustained throughput | High parallel throughput | High compute and bandwidth demand |



## References

[1] J. Cong, B. Liu, S. Neuendorffer, J. Noguera, K. Vissers, and Z. Zhang, "High-Level Synthesis for FPGAs: From Prototyping to Deployment," *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, vol. 30, no. 4, pp. 473–491, 2011.

[2] C. Zhang, P. Li, G. Sun, Y. Guan, B. Xiao, and J. Cong, "Optimizing FPGA-Based Accelerator Design for Deep Convolutional Neural Networks," in *Proceedings of the ACM/SIGDA International Symposium on Field-Programmable Gate Arrays (FPGA)*, Monterey, CA, USA, pp. 161–170, 2015.

[3] Y. Ma, Y. Cao, S. Vrudhula, and J.-S. Seo, "Optimizing the Convolution Operation to Accelerate Deep Neural Networks on FPGA," *IEEE Transactions on Very Large Scale Integration (VLSI) Systems*, vol. 26, no. 7, pp. 1354–1367, 2018.

[4] J. Qiu, J. Wang, S. Yao, K. Guo, B. Li, E. Zhou, J. Yu, T. Tang, N. Xu, S. Song, Y. Wang, and H. Yang, "Going Deeper with Embedded FPGA Platform for Convolutional Neural Network," in *Proceedings of the ACM/SIGDA International Symposium on Field-Programmable Gate Arrays (FPGA)*, pp. 26–35, 2016.
