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

# How DAG Scheduling Actually Works in This VLIW + SIMD CPU

This section explains **how dependency graphs (DAGs)** actually exist inside this code  
and how the simulator CPU executes them **cycle by cycle**.

This is the key to understanding performance optimization here.

---

## First: what part are we analyzing?

We pick **one batch element in one round** from the reference kernel:

```python
idx = mem[inp_indices_p + i]
val = mem[inp_values_p + i]
node_val = mem[forest_values_p + idx]
val = myhash(val ^ node_val)
idx = 2 * idx + (1 if val % 2 == 0 else 2)
idx = 0 if idx >= n_nodes else idx
mem[inp_values_p + i] = val
mem[inp_indices_p + i] = idx
```

This is the inner loop. Everything else is just setup.

---

## Step 1: list the operations (not instructions)

Forget Python. Think in operations:

1. **Load idx**
2. **Load val**
3. **Load node_val**
4. **XOR val ^ node_val**
5. **Hash stage 0**
6. **Hash stage 1**
7. **Hash stage 2**
8. **Hash stage 3**
9. **Hash stage 4**
10. **Hash stage 5**
11. **Compute val % 2**
12. **Compute next idx**
13. **Bounds check**
14. **Store val**
15. **Store idx**

These are the nodes of the DAG.

---

## Step 2: draw the dependency graph (DAG)

Here is the real dependency structure:



```text
Load idx ───────┐
                ├─> Load node_val ─┐
Load val ───────┘                  ├─> XOR ─> Hash0 ─> Hash1 ─> Hash2 ─> Hash3 ─> Hash4 ─> Hash5 ─┐
                                                                                                     ├─> val % 2 ─> select ─┐
                                                                                                     │                      ├─> next idx ─> bounds check ─┐
                                                                                                     │                                               ├─> store idx
                                                                                                     └───────────────────────────────────────────────┘
                                                                                                     └───────────────────────────────────────────────> store val
```

**Key points:**

* Loads must happen before compute
* Hash stages are strictly sequential
* Stores must happen last
* You cannot reorder across arrows

This graph is fixed by logic. Optimization is only about when nodes execute, not changing edges.

---

## Step 3: understand how the CPU executes the DAG

Important CPU rule:

> All reads happen at the start of a cycle  
> All writes commit at the end of a cycle

So:

* You cannot use a value in the same cycle it is produced
* Dependencies force cycle boundaries

---

## Step 4: naive (baseline) scheduling

The baseline kernel does this:

* **Cycle 1:** load idx
* **Cycle 2:** load val
* **Cycle 3:** load node_val
* **Cycle 4:** xor
* **Cycle 5:** hash stage 0
* **Cycle 6:** hash stage 1
* ...
* **Cycle 11:** hash stage 5
* **Cycle 12:** mod
* **Cycle 13:** select
* **Cycle 14:** compute idx
* **Cycle 15:** bounds check
* **Cycle 16:** store val
* **Cycle 17:** store idx

This is correct but terrible. Why? One operation per cycle, most VLIW slots are empty, no overlap, and no SIMD.

---

## Step 5: where the DAG helps optimization

The DAG tells us:

**These can run together:**
* load idx and load val (independent)
* store idx and store val (independent)
* address arithmetic and loads (if inputs ready)

**These cannot:**
* hash stages (strict chain)
* anything that depends on val before hash finishes

So an optimized schedule looks more like:

* **Cycle 1:** load idx, load val
* **Cycle 2:** load node_val, compute address math
* **Cycle 3:** xor, hash stage 0
* **Cycle 4:** hash stage 1, hash stage 2
* **Cycle 5:** hash stage 3, hash stage 4
* **Cycle 6:** hash stage 5, val % 2
* **Cycle 7:** select, compute next idx
* **Cycle 8:** bounds check, store val, store idx

Same DAG. Same result. **Half the cycles.**

---

## Step 6: where SIMD fits in the DAG

SIMD does not change the DAG. It changes how many copies of the DAG run together.

Instead of:
* DAG for element 0
* DAG for element 1
* DAG for element 2

SIMD executes:
* **DAG for elements [0..7] together**

Same arrows. Same dependencies. Just wider.

---

## Final mental model (this is the key)

1. The algorithm defines a **DAG**
2. The DAG defines **what must happen before what**
3. The CPU defines **how much can happen per cycle**

**Optimization is:** executing all ready DAG nodes per cycle without violating dependencies.

---

## One sentence to lock it in

> This code is optimized by scheduling a fixed dependency DAG onto a VLIW CPU so that all independent nodes execute as early as possible within per-cycle hardware limits.

