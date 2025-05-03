# The Poseidon algebraic hash function

## Relative struct or trait

### 1. Domain
```rust
/// A domain which a Poseidon hash function used.
/// F: Field: The finite field used for computations.
/// RATE: usize: The rate of the sponge (number of elements absorbed per round).
pub trait Domain<F: Field, const RATE: usize> {
    /// Iterator that outputs padding field elements.
    type Padding: IntoIterator<Item = F>;

    /// The name of this domain, for debug formatting purposes.
    fn name() -> String;

    /// The initial capacity element, encoding this domain.
    /// TODO: Encodes domain-specific information into the initial state of the Poseidon sponge.
    fn initial_capacity_element() -> F;

    /// Returns the padding to be appended to the input.
    /// TODO: Generates padding to ensure the input length is compatible with the sponge's rate.
    fn padding(input_len: usize) -> Self::Padding;
}
```
`ConstantLength` Struct (Domain specified in [ePrint 2019/458 section 4.2](https://eprint.iacr.org/2019/458.pdf))<br>
The ConstantLength<L> struct is a specific implementation of the Domain trait for constant-length inputs. It is parameterized by:<br>
* `L: usize`: The fixed length of the input.<br>

Implementation of Domain for ConstantLength<L>:<br>
* `name`: Returns a string like "ConstantLength<2>" for debugging.
* `initial_capacity_element`: Encodes the input length L into the sponge's initial state. This ensures that inputs of different lengths result in different hash outputs, even if the input values are the same. Capacity value is $length \cdot 2^{64} + (o-1)$ where o is the output length. We hard-code an output length of 1.
* `padding`: Generates padding to make the input length a multiple of the sponge's rate (RATE). The padding consists of zeroes (F::ZERO).

    for example:<br>
    L = 5, Rate = 3. Step 1: 5 + 3 - 1 = 7, 7 / 3 = 2, k = 2. Step 2: 2 * 3 - 5 = 1, `iter::repeat(F::ZERO).take(1)` generates a padding of one zero element. This ensures that the input length is compatible with the sponge's rate.
```rust
#[derive(Clone, Copy, Debug)]
pub struct ConstantLength<const L: usize>;

impl<F: PrimeField, const RATE: usize, const L: usize> Domain<F, RATE> for ConstantLength<L> {
    type Padding = iter::Take<iter::Repeat<F>>;

    fn name() -> String {
        format!("ConstantLength<{}>", L)
    }

    fn initial_capacity_element() -> F {
        // Capacity value is $length \cdot 2^64 + (o-1)$ where o is the output length.
        // We hard-code an output length of 1.
        F::from_u128((L as u128) << 64)
    }

    fn padding(input_len: usize) -> Self::Padding {
        assert_eq!(input_len, L);
        // For constant-input-length hashing, we pad the input with zeroes to a multiple
        // of RATE. On its own this would not be sponge-compliant padding, but the
        // Poseidon authors encode the constant length into the capacity element, ensuring
        // that inputs of different lengths do not share the same permutation.
        let k = (L + RATE - 1) / RATE;
        iter::repeat(F::ZERO).take(k * RATE - L)
    }
}
```

### 2. Spec
The Hades design strategy, used in the Poseidon hash function, is a cryptographic approach that optimizes the trade-off between security and efficiency. It incorporates a mix of full and partial rounds within the hash function.<br>
* Full rounds apply a non-linear operation (like an S-box) to all elements of the state, offering high security.
* Partial rounds, on the other hand, apply this operation to only a subset of the state elements, enhancing efficiency.

This combination allows for maintaining strong cryptographic security while reducing the computational overhead typically associated with full rounds in every step. This design is particularly beneficial in contexts like zero-knowledge proofs, where efficiency is crucial without compromising security. The paper highlights how the choice of the MDS matrix in HADES significantly affects security against differential and linear attacks.

Generics<br>
| Type | Description |
|------|-------------|
| `F: Field` | a. F is the finite field type used for computations in the Poseidon permutation.<br> b. The Field trait ensures that F supports basic arithmetic operations like addition, multiplication, and inversion. |
| `const T: usize` | a. T is the width of the Poseidon state, i.e., the number of field elements in the state array.<br> b. It determines the size of the internal state used during the permutation. |
| `const RATE: usize` | a. RATE is the sponge rate, which specifies how many elements can be absorbed (TODO:) or squeezed in each round of the sponge construction.<br> b. It must satisfy RATE < T because some elements in the state are reserved for the sponge's capacity. |

The `Spec` trait provides a blueprint for configuring the Poseidon permutation. It defines:<br>
* The number of full and partial rounds.
* The S-box for non-linear transformations.
* The MDS matrix for mixing state elements.
* The round constants for each round.

By implementing this trait, you can define a specific Poseidon permutation tailored to your cryptographic requirements.
```rust
/// A specification for a Poseidon permutation.
pub trait Spec<F: Field, const T: usize, const RATE: usize>: fmt::Debug {
    /// The number of full rounds for this specification.
    ///
    /// This must be an even number.
    fn full_rounds() -> usize;

    /// The number of partial rounds for this specification.
    fn partial_rounds() -> usize;

    /// The S-box for this specification.
    fn sbox(val: F) -> F;

    /// Side-loaded index of the first correct and secure MDS that will be generated by
    /// the reference implementation.
    ///
    /// This is used by the default implementation of [`Spec::constants`]. If you are
    /// hard-coding the constants, you may leave this unimplemented.
    fn secure_mds() -> usize;

    /// Generates `(round_constants, mds, mds^-1)` corresponding to this specification.
    fn constants() -> (Vec<[F; T]>, Mds<F, T>, Mds<F, T>);
}
```

 Associated Methods<br>
| Method | Description |
|--------|-------------|
| `fn full_rounds() -> usize` | a. Returns the number of full rounds in the Poseidon permutation.<br> b. A full round applies the S-box (non-linear transformation) to all elements in the state, followed by a matrix multiplication (MDS matrix).<br> c. The number of full rounds must be even to ensure symmetry and security. |
| `fn partial_rounds() -> usize` | a. Returns the number of partial rounds in the Poseidon permutation.<br> b. In a partial round, the S-box is applied to (TODO:) only one element of the state (usually the first element), followed by a matrix multiplication.<br> c. Partial rounds are used to reduce computational cost while maintaining security. |
| `fn sbox(val: F) -> F` | a. Defines the S-box (substitution box), which is a non-linear transformation applied to elements of the state.<br> b. The S-box is a critical component for ensuring the cryptographic security of the permutation.<br> c. Common choices for the S-box include raising the element to a power (e.g., x^5). |
| `fn secure_mds() -> usize` | a. Returns the index of the first secure MDS matrix generated by the reference implementation.<br> b. The MDS (Maximum Distance Separable) matrix is used for mixing the state elements during the permutation.<br> c. This method is optional and is primarily used when the MDS matrix is generated dynamically. If the constants are hard-coded, this method can be left unimplemented. |
| `fn constants() -> (Vec<[F; T]>, Mds<F, T>, Mds<F, T>)` | Generates the constants required for the Poseidon permutation:<br> a. `round_constants`:<br> A vector of arrays, where each array contains the round constants for a single round.<br> Round constants are added to the state elements during each round to ensure security.<br> b. `mds`:<br> The MDS matrix, which is used for mixing the state elements.<br> c. `mds^-1`:<br> The inverse of the MDS matrix, which may be used in certain applications.

`generate_constants` Method<br>
The generate_constants function is responsible for generating the constants required for the Poseidon permutation. These constants include:
* Round Constants: Used in each round of the Poseidon permutation to ensure cryptographic security.
* MDS Matrix: A mixing matrix that ensures diffusion across the state.
* Inverse MDS Matrix: The inverse of the MDS matrix, which may be used in certain applications.

This function is generic over the field type F, the Poseidon specification S, and the parameters T (state width) and RATE (sponge rate).

Generics<br>
| Type | Description |
|------|-------------|
| `F: FromUniformBytes<64> + Ord` | a. F is the finite field type used for computations.<br> b. FromUniformBytes<64> ensures that F can be constructed from 64 bytes of uniform randomness, which is required for generating random field elements.<br> c. Ord ensures that F supports ordering, which may be required for certain operations. |
| `S: Spec<F, T, RATE>` | S is the Poseidon specification, which defines parameters like the number of rounds, the S-box, and the MDS matrix. | 
| `const T: usize` | T is the width of the Poseidon state, i.e., the number of field elements in the state. |
| `const RATE: usize` | RATE is the sponge rate, which specifies how many elements can be absorbed or squeezed in each round of the sponge construction. |

Return Value<br>
| Type | Description |
|------|-------------|
| `Vec<[F; T]>` | A vector of round constants, where each element is an array of T field elements. | 
| `Mds<F, T>` | The MDS matrix, represented as a 2D array of size T x T. |
| `Mds<F, T>` | The inverse of the MDS matrix. |

Function Logic<br>
| Step | Description |
|------|-------------|
| 1. Retrieve Round Parameters | a. r_f: The number of full rounds, retrieved from the Poseidon specification S.<br> b. r_p: The number of partial rounds, also retrieved from S. |
| 2. Initialize the Grain Generator | The Grain generator is used to produce random field elements for the round constants and MDS matrix. | 
| 3. Generate Round Constants | a. A total of `r_f` + `r_p` rows of round constants are generated, where each row contains `T` field elements.<br> b. For each row: An array `rc_row` of size T is initialized with zeros. Each element of `rc_row` is replaced with a random field element generated by `grain.next_field_element()`.<br> c. The rows are collected into a vector round_constants. |
| 4. Generate MDS Matrix and Its Inverse | a. The `generate_mds` function is called to generate the MDS matrix and its inverse.<br> b. The grain generator is used to produce the random elements required for the MDS matrix.<br> c. `S::secure_mds()` provides the index of the first secure MDS matrix, as defined by the Poseidon specification. |
| 5. Return the Constants | The function returns the round constants, MDS matrix, and inverse MDS matrix as a tuple. |

```rust
/// Generates `(round_constants, mds, mds^-1)` corresponding to this specification.
pub fn generate_constants<
    F: FromUniformBytes<64> + Ord,
    S: Spec<F, T, RATE>,
    const T: usize,
    const RATE: usize,
>() -> (Vec<[F; T]>, Mds<F, T>, Mds<F, T>) {
    let r_f = S::full_rounds();
    let r_p = S::partial_rounds();

    let mut grain = grain::Grain::new(SboxType::Pow, T as u16, r_f as u16, r_p as u16);

    let round_constants = (0..(r_f + r_p))
        .map(|_| {
            let mut rc_row = [F::ZERO; T];
            for (rc, value) in rc_row
                .iter_mut()
                .zip((0..T).map(|_| grain.next_field_element()))
            {
                /*
                    1. rc is a mutable reference to an element in the array rc_row.
                    2. value is a field element generated by grain.next_field_element().
                    3. The line *rc = value; assigns the value of value to the element in rc_row that rc refers to.
                */
                *rc = value;
            }
            rc_row
        })
        .collect();

    let (mds, mds_inv) = mds::generate_mds::<F, T>(&mut grain, S::secure_mds());

    (round_constants, mds, mds_inv)
}
```

### 2. Sponge
[(Wikipedia)](https://en.wikipedia.org/wiki/Sponge_function)In cryptography, a sponge function or sponge construction is any of a class of algorithms with finite internal state that take an input bit stream of any length and produce an output bit stream of any desired length. Sponge functions have both theoretical and practical uses. They can be used to model or implement many cryptographic primitives, including cryptographic hashes, message authentication codes, mask generation functions, stream ciphers, pseudo-random number generators, and authenticated encryption.<sup>[[1]](https://eprint.iacr.org/2019/458)</sup>

The sponge construction for hash functions. $P_i$ are blocks of the input string, $Z_i$ are hashed output blocks.
![alt text](sponge-function.png)

Like other hash algorithms, Poseidon uses the sponge construction. Sponges are a general method for reading in a data stream at a certain rate, allowing for mixing of the data within the capacity of the sponge, and then outputting a fixed-size hash of the data. There are two important parameters used to construct the sponge that impact the security of the hash function:<br>
* The rate $r$: Determines how many chunks of size $|𝔽_p|$ are "absorbed" into the sponge in each step.
* The capacity: Determines how many chunks of size $|𝔽_p|$ are stored in the sponge in each step.<br>

TODO: The higher the rate, the faster and cheaper the hash becomes, but this also makes the hash less secure. Intuitively, the larger the capacity, the more random state you allow yourself to store in the sponge in each step.<br>

The `Sponge` struct is the core of the Poseidon sponge construction. It encapsulates the state, mode, and parameters required for the Poseidon permutation.<br>
| Field | Description |
|-------|-------------|
| `mode: M` | a. Represents the current mode of the sponge (either `Absorbing` or `Squeezing`).<br> b. Determines whether the sponge is absorbing input or squeezing output. |
| `state: State<F, T>` | a. The internal state of the sponge, represented as an array of `T` field elements.<br> b. This state is updated during the Poseidon permutation. |
| `mds_matrix: Mds<F, T>` | a. The Maximum Distance Separable (MDS) matrix used in the Poseidon permutation.<br> b. Ensures diffusion across the state during the permutation. |
| `round_constants: Vec<[F; T]>` | a. A vector of round constants used in the Poseidon permutation.<br> b. These constants are added to the state during each round of the permutation. |
| `_marker: PhantomData<S>` | a. A marker for the Poseidon specification type `S`.<br> b. Used to associate the sponge with a specific Poseidon parameterization without storing actual data. |
```rust
pub(crate) struct Sponge<
    F: Field,
    S: Spec<F, T, RATE>,
    M: SpongeMode,
    const T: usize,
    const RATE: usize,
> {
    mode: M,
    state: State<F, T>,
    mds_matrix: Mds<F, T>,
    round_constants: Vec<[F; T]>,
    _marker: PhantomData<S>,
}
```
The Sponge (`State`)<br>
Think of the sponge as having two action: it first absorbs input data like a sponge absorbs liquid, and subsequently this data gets squeezed to produce output data like a sponge releases liquid. In cryptographic terms, this sponge is the internal state of the hash function.<br>
| Phase | Description |
|-------|-------------|
| 1. Absorbing Phase | a. Data such as a message, a file, or a number is divided into blocks, for instance, of a given number of bits or the size of a finite field.<br> b. Each block of your input data is "absorbed" into the sponge where it is then mixed with the residual data (like the residual "liquid") in the sponge. The absorption and mixing process are defined concretely by the rate and capacity of the sponge respectively. This mixing process can be thought of like absorbing some new blue colored water into one chunk of the sponge and some new red in another while there is already some green soaking in the rest of the sponge. Fitting to the analogy, to truly mix these, the sponge would be allowed to sit where diffusion of the colors would occur. In the cryptographic case, we use a permutation to mix the data like diffusion can mix the colors. | 
| 2. Squeezing Phase | a. Once the sponge has absorbed all your data (or all the colored water in the analogy), it's time to get the hash output, similar to squeezing water out of the sponge.<br> b. In the squeezing phase, part of the sponge's state is read out as the hash result. If the desired hash output is longer, the sponge might be "squeezed" multiple times, with additional transformations in between to ensure security. If it is shorter, the sponge is squeezed only once, and the output is truncated to the desired length. By squeezing the sponge, we are essentially wringing out the water, leaving us with a fixed amount of water (the hash) that is a mix of all the colors (the input data). The blue, red, and green water in our analogy now becomes a single murky brownish purplish color, which is the hash output. Could you ever get the original colors back from this let alone the order they were added in? |

`SealedSpongeMode`<br>
```rust
mod private {
    pub trait SealedSpongeMode {}
    impl<F, const RATE: usize> SealedSpongeMode for super::Absorbing<F, RATE> {}
    impl<F, const RATE: usize> SealedSpongeMode for super::Squeezing<F, RATE> {}
}
```
| Detail | Description |
|--------|-------------|
| Purpose | a. `SealedSpongeMode` is a private trait used to seal the `SpongeMode` trait.<br> b. This ensures that only the Absorbing and Squeezing types (defined in this module) can implement SpongeMode. |
| Why Sealing? | Sealing prevents external types from implementing `SpongeMode`, which ensures that the sponge's state transitions are controlled and predictable. | 

`SpongeMode`<br>
```rust
/// The state of the `Sponge`.
pub trait SpongeMode: private::SealedSpongeMode {}
```
| Detail | Description |
|--------|-------------|
| Purpose | a. `SpongeMode` is a public trait that represents the state of the sponge.<br> b. It is implemented by the Absorbing and Squeezing types. | 
| Relationship with `SealedSpongeMode` | `SpongeMode` inherits from `SealedSpongeMode`, so only types that implement `SealedSpongeMode` (i.e., `Absorbing` and `Squeezing`) can implement SpongeMode. |

`Absorbing`<br>
```rust
/// The absorbing state of the `Sponge`.
#[derive(Debug)]
pub struct Absorbing<F, const RATE: usize>(pub(crate) SpongeRate<F, RATE>);

/// This allows Absorbing to be treated as a valid sponge state.
impl<F, const RATE: usize> SpongeMode for Absorbing<F, RATE> {}
```
| Detail | Description |
|--------|-------------|
| Purpose | a. Represents the absorbing state of the sponge.<br> b. The sponge is in this state when it is absorbing input values. |
| Fields | `SpongeRate<F, RATE>`: An array of size RATE that holds the absorbed values. Each element is an Option<F> to indicate whether it is occupied. |

`Squeezing`<br>
```rust
/// The squeezing state of the `Sponge`.
#[derive(Debug)]
pub struct Squeezing<F, const RATE: usize>(pub(crate) SpongeRate<F, RATE>);

/// This allows Squeezing to be treated as a valid sponge state.
impl<F, const RATE: usize> SpongeMode for Squeezing<F, RATE> {}
```
| Detail | Description |
|--------|-------------|
| Purpose | a. Represents the squeezing state of the sponge.<br> b. The sponge is in this state when it is extracting (squeezing) output values. |
| Fields | `SpongeRate<F, RATE>`: An array of size RATE that holds the squeezed values. Each element is an Option<F> to indicate whether it is available for extraction. |

### 3. Hash
```rust
/// A Poseidon hash function, built around a sponge.
pub struct Hash<
    /// This trait represents an element of a field, which is used for computations.
    F: Field,
    /// A specification for a Poseidon permutation.
    S: Spec<F, T, RATE>,
    /// A domain which a Poseidon hash function used.
    D: Domain<F, RATE>,
    /// TODO: The width of the Poseidon state (number of field elements in the state).
    const T: usize,
    /// TODO: The rate of the sponge (number of elements absorbed/squeezed per round).
    const RATE: usize,
> {
    /// A Poseidon sponge.
    /// Absorbing: The absorbing state of the `Sponge`.
    sponge: Sponge<F, S, Absorbing<F, RATE>, T, RATE>,
    /*
        (Rust Syntax)

        In the given code, PhantomData<D> is a marker used in Rust to indicate that the Hash struct logically depends on the generic type parameter D, even though D is not directly used as a field in the struct. This is a common pattern in Rust to express ownership or type relationships without storing actual data of that type.

        PhantomData<D> does not occupy any memory in the struct. It is purely a compile-time marker.
    */
    /// A marker for the domain type.
    _domain: PhantomData<D>,
}
```
The `Debug` implementation provides a way to print the internal state of the Hash struct for debugging purposes. It includes:<br>
* width (T): The width of the Poseidon state.
* rate (RATE): The rate of the sponge.
* R_F: The number of full rounds in the Poseidon permutation.
* R_P: The number of partial rounds in the Poseidon permutation.
* domain: The name of the domain.

`Hash::init` Method initializes a new Poseidon hasher. It:<br>
* Creates a new Sponge with an initial capacity element provided by the domain.
* Sets the _domain marker to the appropriate type.
```rust
impl<F: Field, S: Spec<F, T, RATE>, D: Domain<F, RATE>, const T: usize, const RATE: usize>
    Hash<F, S, D, T, RATE>
{
    /// Initializes a new hasher.
    pub fn init() -> Self {
        Hash {
            sponge: Sponge::new(D::initial_capacity_element()),
            _domain: PhantomData::default(),
        }
    }
}
```
`Hash::hash` Method hashes a fixed-length input message. It is implemented for the ConstantLength<L> domain (TODO: Domain specified in [ePrint 2019/458 section 4.2](https://eprint.iacr.org/2019/458.pdf)), where L is the length of the input message. The steps are:<br>
* The input message is iterated over using .into_iter().
* Padding is appended to the input using `<ConstantLength<L> as Domain<F, RATE>>::padding(L)`. This ensures that the input length is a multiple of the sponge's RATE.
* Each value (from the message and padding) is absorbed into the sponge using `self.sponge.absorb(value)`. If the sponge's state is full, (TODO:)it triggers a Poseidon permutation to process the absorbed values.
* After all input values are absorbed, the sponge transitions from the `absorbing` state to the `squeezing` state by calling `self.sponge.finish_absorbing()`. This finalizes the internal state of the sponge.
* The sponge produces the hash output by calling `self.sponge.squeeze()`. This extracts a single field element from the sponge's state.
```rust
impl<F: PrimeField, S: Spec<F, T, RATE>, const T: usize, const RATE: usize, const L: usize>
    Hash<F, S, ConstantLength<L>, T, RATE>
{
    /// Hashes the given input.
    /// Input: A fixed-size array of L field elements to be hashed.
    /// Output: Returns a single field element F as the hash result.
    pub fn hash(mut self, message: [F; L]) -> F {
        for value in message
            .into_iter()
            .chain(<ConstantLength<L> as Domain<F, RATE>>::padding(L))
        {
            self.sponge.absorb(value);
        }
        self.sponge.finish_absorbing().squeeze()
    }
}
```

# References
- [Poseidon hash function](https://www.poseidon-hash.info/)
- [Poseidon Journal](https://autoparallel.github.io/overview/index.html)
- [Grain LSFR](https://autoparallel.github.io/poseidon/round_constants.html?highlight=grain#grain-lsfr)
