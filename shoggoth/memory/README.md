# High-level overview

This directory contains all the major building blocks for our memory
implementation.  The memory is borrowing from Reverb nomenclature, which in turn
is borrowing from databases. What is not alike Reverb is that we do not have the
RateSamplers (but it can be added). We also do not share data between Tables.

The goal of the memory is to provide a unified interface for all types of
machine learning tasks. This is achieved by focusing on configuration and
pluggability over code-driven functionality.

Currently, there are three main points of customization:

* Shape and type of data
* Insertion, sampling, and eviction
* Data transformation and generation

## High-level parts

### Table
A table is a datastructure containing a specific type of data that shares the same high-level structure.

### Column
A column is a storage for a specific type of data where each item is the same shape and type

### VirtualColumn
A virtual column is like a column, but it references another column and does
data synthesization or modification w.r.t that. For example, dones and masks are
synthetic data based only on indices.

### Adaptors

Adaptors are another approach to virtual column but are more suited for
transforming the whole batch, such as scaling for reshaping specific
datas. Since this step occurs when the data has already been converted to
tensors, the full power of Tensorflow is available here and gradients will be
correctly tracked.

### Strategies, Samplers and Ejectors
Strategies are based on the delegate pattern, where we can inject implementation
details through objects instead of using inheritance. Strategies define the API
for sampling and ejection from memories, and are queried from the table upon
sampling and insertion.

Samplers and Ejectors track the data (but do not own it!). They are used by the
table for sampling and ejection based on the policy they implement. Currently we
have Fifo and Uniform samplers and ejectors, but one could have prioritized
samplers/ejectors, etc.
