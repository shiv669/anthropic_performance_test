# Performance Engineering Take-Home – What This Repo Is Actually About

This repo is **not about trees**, **not about hashing**, and **not about Python**.

This repo is about **performance**.

More specifically:
👉 how to run the *same exact computation* in **fewer CPU cycles** on a **fake VLIW + SIMD processor**.

The simulator is just a tool.  
The real problem is **instruction scheduling**.

---

## What are we trying to do?

We are given:
- a **fixed algorithm** (tree traversal + hashing)
- a **fixed CPU model** (VLIW + SIMD with strict limits)
- a **baseline kernel** that is intentionally slow

Our task is:
> Produce the same final memory result  
> but using **as few cycles as possible**.

Performance is measured only by:
machine.cycle

Lower cycles = better solution.

Nothing else matters.

---

## What kind of CPU is being simulated?

This simulator models a simplified **AI-style accelerator**, not a normal CPU.

Key properties:

- **VLIW (Very Large Instruction Word)**  
  One instruction = many operations in parallel

- **SIMD (Vector execution)**  
  One instruction operates on multiple numbers at once  
  (`VLEN = 8`)

- **Explicit memory and registers**  
  No caches. No guessing. No reordering.

- **Hard per-cycle limits**  

  Per cycle, at most:
  - 12 scalar ALU ops
  - 6 vector ALU ops
  - 2 loads
  - 2 stores
  - 1 control-flow op

The CPU does not “think”.
It just executes whatever the compiler scheduled.

---

## What is the algorithm doing (in simple words)?

Ignore the simulator for a moment.

Logically, the algorithm does this:

For each round  
For each element in a batch:
1. Read the current tree index
2. Read the current value
3. Read the tree node value
4. Mix value with node value using a hash
5. Decide left or right based on the result
6. Update index and value back to memory

That’s it.

Everything else in this repo exists only to run this efficiently.

---

## What does “optimize” mean here?

**Optimize does NOT mean:**
- changing the algorithm
- using smarter math
- skipping work

**Optimize DOES mean:**
- doing independent work in parallel
- overlapping loads, compute, and stores
- using SIMD instead of scalar ops
- avoiding wasted cycles

The algorithm stays identical.
Only the **schedule** changes.

---

## Picking ONE inner loop to analyze

Let’s focus on **one batch element in one round**.

This is the inner loop that dominates runtime.

### Required logical work (cannot be removed)

For one element, we must do:

- 2 memory loads (index, value)
- 1 memory load (tree node)
- ~6 hash stages (pure arithmetic)
- index update arithmetic
- 2 memory stores (index, value)

These operations are **mandatory**.

---

## Theoretical minimum cycles (best possible case)

Now think like hardware.

Per cycle limits:
- Load: 2
- Store: 2
- ALU: 12

### Minimum cycles needed (lower bound)

Even with perfect scheduling:

- Loads:  
  3 loads total → minimum **2 cycles**
- Stores:  
  2 stores total → minimum **1 cycle**
- Arithmetic:  
  Hash + index math needs multiple ALU ops → **at least 1–2 cycles**
- Control flow:  
  selection logic costs **1 flow slot**

👉 Absolute lower bound:  
**~4–5 cycles per element**

This is the *physics limit* of the simulator.

You cannot beat this.

---

## What the baseline kernel actually does

The baseline kernel does **one operation per cycle**.

That means:
- load → cycle
- add → cycle
- xor → cycle
- compare → cycle
- store → cycle

No overlap.  
No parallelism.  
No SIMD.

So instead of ~5 cycles, it uses **dozens of cycles** per element.

This is intentional.
The baseline is meant to be bad.

---

## Where cycles are wasted (concrete)

Major sources of waste:

1. **No slot packing**  
   Independent operations run in separate cycles.

2. **No SIMD usage**  
   Batch elements processed one-by-one instead of 8 at once.

3. **Redundant address calculations**  
   Same addresses recomputed multiple times.

4. **Memory not overlapped with compute**  
   Loads, compute, and stores are serialized.

5. **Too many control-flow instructions**  
   Flow unit becomes a bottleneck.

---

## What a good optimization does

A good optimized kernel:

- Packs multiple ALU + load + store ops into one cycle
- Processes **8 batch elements at once** using SIMD
- Reuses scratch values instead of recomputing
- Hides memory latency behind arithmetic
- Keeps hardware units busy every cycle

The goal is not to be clever.
The goal is to **fill the slots**.

---

## Final mental model

Think of each cycle like a box:
Cycle: [ load ] [ load ] [ alu ] [ alu ] [ alu ] ... [ store ] [ store ]

If boxes are empty → performance is bad.

Optimization is simply:
> moving work around so fewer boxes are empty.

---
