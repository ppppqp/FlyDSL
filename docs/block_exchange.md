# Cooperative block exchange

`fx.coop.BlockExchange[dtype, block_size, items_per_thread]` redistributes a
register-held tile between named thread/value arrangements. It uses static Fly
layouts to compute logical ownership. Identity conversions stay in registers,
wave-local conversions use packed lane moves, and cross-wave conversions use
padded shared memory.

```python
exchange = fx.coop.BlockExchange[fx.Float32, 256, 4]
storage = fx.SharedAllocator().allocate(exchange.SharedStorage).peek()

# values is a flat Vector of four Float32 values per thread.
striped = exchange.blocked_to_striped(values, storage=storage)
fx.barrier()  # Finish reads before reusing the same storage.
blocked = exchange.striped_to_blocked(striped, storage=storage)
fx.barrier()
warp_striped = exchange.blocked_to_warp_striped(blocked)  # No LDS needed.
blocked_again = exchange.warp_striped_to_blocked(warp_striped)
```

For thread `t`, item `i`, block size `B`, items per thread `I`, and logical
warp width `W`, the arrangements assign these logical element indices:

| Arrangement | Logical element |
| --- | --- |
| Blocked | `t * I + i` |
| Striped | `t + i * B` |
| Warp-striped | `(t // W) * W * I + t % W + i * W` |

`block_size` accepts an integer or `(x, y, z)`, with x varying fastest. Its
product must be a power of two no larger than 1024; `items_per_thread` must
also be a positive power of two. The logical warp width is the smaller of the
block size and the target's hardware wave size, including for partial waves.
The input must be a flat Vector with exactly the specialized dtype and length.
Numeric element widths of 8, 16, 32, and 64 bits are supported.

Every block thread must participate together, and the specialization must match
the launch shape. Cross-wave exchanges perform a barrier between shared-memory
stores and loads, with no trailing barrier: callers must synchronize before
reusing that storage. Wave-local and identity conversions need no storage or
block barriers. Passing existing storage remains valid on all paths.
`fx.coop.universal.BlockExchange` uses portable GPU shuffles for its lane moves.

The shared buffer shifts every 32 groups by one group, where a group holds at
least four bytes or one element. This padding preserves correctness across
bank geometries but does not guarantee conflict-free accesses on every target.
General user-supplied layouts remain follow-up work.

## Whole-vector lane permutation

`fx.coop.warp_permute(value, source_lane, width=None)` reads the scalar or whole
Vector held by another active lane. `source_lane` is relative to the current
aligned logical group, with a power-of-two `width` (hardware wave size by
default). Every requested source lane must be active and execute the call.
No block barrier is performed. Partial physical waves are supported as long as
each requested source lane is active.

The AMD lowering packs 8- and 16-bit elements into 32-bit words, pads a partial
last word, and splits 64-bit elements into two words. Each word uses one
`ds_bpermute`; the result retains the original bits, dtype, and vector shape.
The portable spelling, `fx.coop.universal.warp_permute`, uses GPU shuffles.
`fx.coop.warp_permute_xor(value, lane_mask, width=None)` handles constant XOR
pairings. Masks below 32 use a packed `ds_swizzle`; masks that cross wave64's
halves use `ds_bpermute`. The fused RoPE kernel uses this XOR form for its
rotary pairs.

BlockExchange routes its named permutations by ownership: one item per thread
or one thread is an identity, warp striping always stays within a wave, and
block striping stays within a wave only for a single-wave block. The current
wave gather can issue multiple lane moves per output item; it is not a tuned
register-transpose algorithm for large items-per-thread counts.

Validation lives in `tests/extension/coop/test_block_exchange.py`: GPU checks
compare each arrangement against host tensor transposes across element types,
item counts, block shapes, and all power-of-two widths from 1 to 1024. A round
trip checks shared-storage reuse with the required barrier. Host-side checks
cover specialization and input diagnostics.
